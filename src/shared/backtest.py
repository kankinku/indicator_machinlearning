from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import config


@dataclass
class BacktestResult:
    dates: List[str]
    equity_pct: List[float]
    trades: List[Dict[str, object]]
    total_return_pct: float
    win_rate: float
    trade_count: int
    valid_trade_count: int  # [V11.2] Regime-aligned trades
    mdd_pct: float


def _coerce_float(value: object, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def derive_trade_params(risk_budget: Dict[str, object]) -> Tuple[float, float, int]:
    sl_pct = _coerce_float(risk_budget.get("sl_pct"))
    tp_pct = _coerce_float(risk_budget.get("tp_pct"))
    if sl_pct is None:
        sl_pct = _coerce_float(risk_budget.get("stop_loss"))

    if tp_pct is None:
        rr = _coerce_float(risk_budget.get("risk_reward_ratio"))
        if sl_pct is not None and rr is not None:
            tp_pct = sl_pct * rr

    k_up = _coerce_float(risk_budget.get("k_up"))
    k_down = _coerce_float(risk_budget.get("k_down"))
    est_vol = config.RISK_EST_DAILY_VOL
    if tp_pct is None and k_up is not None:
        tp_pct = k_up * est_vol
    if sl_pct is None and k_down is not None:
        sl_pct = k_down * est_vol

    if tp_pct is None:
        tp_pct = 0.02
    if sl_pct is None:
        sl_pct = 0.01

    tp_pct = max(tp_pct, 0.001)
    sl_pct = max(sl_pct, 0.001)

    horizon = risk_budget.get("horizon")
    try:
        horizon = int(horizon) if horizon is not None else 20
    except (TypeError, ValueError):
        horizon = 20
    horizon = max(horizon, 1)
    return tp_pct, sl_pct, horizon


def _extract_signal_index(results_df: Optional[pd.DataFrame]) -> Optional[pd.DatetimeIndex]:
    if results_df is None or results_df.empty:
        return None
    if "date" in results_df.columns:
        idx = pd.to_datetime(results_df["date"], errors="coerce")
        idx = idx.dropna()
        if not idx.empty:
            return pd.DatetimeIndex(idx)
        return None
    if isinstance(results_df.index, pd.DatetimeIndex):
        return results_df.index
    return None


def build_signal_series(results_df: Optional[pd.DataFrame], price_df: pd.DataFrame) -> pd.Series:
    if results_df is None or results_df.empty:
        return pd.Series(0, index=price_df.index, dtype=int)
    df = results_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
    if "pred" in df.columns:
        signal = df["pred"]
    elif "signal" in df.columns:
        signal = df["signal"]
    else:
        return pd.Series(0, index=price_df.index, dtype=int)
    signal = signal.reindex(price_df.index).fillna(0)
    return signal.astype(int)


def simulate_backtest(
    price_df: pd.DataFrame,
    signal_series: pd.Series,
    tp_pct: float,
    sl_pct: float,
    horizon: int,
    cost_bps: float,
    target_regime: Optional[str] = None,  # [V11.2]
) -> BacktestResult:
    if price_df.empty:
        return BacktestResult([], [], [], 0.0, 0.0, 0, 0, 0.0)

    df = price_df.copy()
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            df[col] = df["close"]
    df = df[["open", "high", "low", "close"]]

    signals = signal_series.reindex(df.index).fillna(0).astype(int)
    dates = df.index

    cost_pct = (2 * float(cost_bps)) / 10000.0
    equity = 1.0
    equity_curve: List[float] = []
    trades: List[Dict[str, object]] = []

    pending_dir = 0
    pending_idx: Optional[int] = None
    position = 0
    entry_price: Optional[float] = None
    entry_idx: Optional[int] = None
    hold_bars = 0

    for i in range(len(df)):
        signal = int(signals.iloc[i])

        if pending_dir != 0 and pending_idx == i:
            position = pending_dir
            entry_price = float(df["open"].iloc[i])
            entry_idx = i
            hold_bars = 0
            pending_idx = i + 1
            
        # [V11.2] Regime Check for Entry (if provided)
        # Entry price is usually open of next bar, so we check regime at entry_idx or pending_idx
        is_regime_match = True
        if target_regime and "regime_label" in df.columns:
            # Check the regime at the bar where signal was generated
            is_regime_match = df["regime_label"].iloc[i] == target_regime

        if position == 0:
            if signal != 0 and i + 1 < len(df):
                pending_dir = signal
                pending_idx = i + 1
            equity_curve.append((equity - 1.0) * 100.0)
            continue

        hold_bars += 1
        high = float(df["high"].iloc[i])
        low = float(df["low"].iloc[i])
        close = float(df["close"].iloc[i])

        exit_price = None
        exit_reason = None

        if position == 1:
            target_price = entry_price * (1.0 + tp_pct)
            stop_price = entry_price * (1.0 - sl_pct)
            hit_target = high >= target_price
            hit_stop = low <= stop_price
            if hit_target and hit_stop:
                exit_price = stop_price
                exit_reason = "SL"
            elif hit_target:
                exit_price = target_price
                exit_reason = "TP"
            elif hit_stop:
                exit_price = stop_price
                exit_reason = "SL"
        else:
            target_price = entry_price * (1.0 - tp_pct)
            stop_price = entry_price * (1.0 + sl_pct)
            hit_target = low <= target_price
            hit_stop = high >= stop_price
            if hit_target and hit_stop:
                exit_price = stop_price
                exit_reason = "SL"
            elif hit_target:
                exit_price = target_price
                exit_reason = "TP"
            elif hit_stop:
                exit_price = stop_price
                exit_reason = "SL"

        if exit_price is None and hold_bars >= horizon:
            exit_price = close
            exit_reason = "HORIZON"

        if exit_price is None and (signal == 0 or signal != position):
            exit_price = close
            exit_reason = "AGENT_EXIT"

        if exit_price is not None:
            pnl_pct = (exit_price / entry_price - 1.0) * position
            pnl_pct -= cost_pct
            equity *= (1.0 + pnl_pct)
            trades.append({
                "entry_date": str(dates[entry_idx].date()),
                "exit_date": str(dates[i].date()),
                "direction": position,
                "entry_price": round(entry_price, 6),
                "exit_price": round(exit_price, 6),
                "return_pct": round(pnl_pct * 100.0, 4),
                "reason": exit_reason,
                "bars": hold_bars,
                "is_valid": is_regime_match, # [V11.2]
            })
            position = 0
            entry_price = None
            entry_idx = None
            hold_bars = 0

        equity_curve.append((equity - 1.0) * 100.0)

    if position != 0 and entry_price is not None and entry_idx is not None:
        close = float(df["close"].iloc[-1])
        pnl_pct = (close / entry_price - 1.0) * position
        pnl_pct -= cost_pct
        equity *= (1.0 + pnl_pct)
        trades.append({
            "entry_date": str(dates[entry_idx].date()),
            "exit_date": str(dates[-1].date()),
            "direction": position,
            "entry_price": round(entry_price, 6),
            "exit_price": round(close, 6),
            "return_pct": round(pnl_pct * 100.0, 4),
            "reason": "HORIZON",
            "bars": hold_bars,
            "is_valid": is_regime_match,
        })
        if equity_curve:
            equity_curve[-1] = (equity - 1.0) * 100.0
        else:
            equity_curve.append((equity - 1.0) * 100.0)

    trade_returns = [t["return_pct"] for t in trades]
    wins = sum(1 for v in trade_returns if v > 0)
    trade_count = len(trade_returns)
    valid_trade_count = sum(1 for t in trades if t.get("is_valid", True))
    win_rate = wins / trade_count if trade_count > 0 else 0.0

    equity_arr = np.array([1.0 + v / 100.0 for v in equity_curve]) if equity_curve else np.array([1.0])
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr / peak) - 1.0
    mdd_pct = abs(float(np.min(drawdown)) * 100.0)

    date_str = [d.strftime("%Y-%m-%d") for d in dates]
    total_return_pct = (equity - 1.0) * 100.0

    return BacktestResult(
        dates=date_str,
        equity_pct=equity_curve,
        trades=trades,
        total_return_pct=total_return_pct,
        win_rate=win_rate,
        trade_count=trade_count,
        valid_trade_count=valid_trade_count,
        mdd_pct=mdd_pct,
    )


def run_signal_backtest(
    price_df: pd.DataFrame,
    results_df: Optional[pd.DataFrame],
    risk_budget: Dict[str, object],
    cost_bps: float,
    target_regime: Optional[str] = None,
) -> BacktestResult:
    signal_idx = _extract_signal_index(results_df)
    if signal_idx is not None and not signal_idx.empty:
        start = signal_idx.min()
        end = signal_idx.max()
        price_df = price_df.loc[start:end]

    tp_pct, sl_pct, horizon = derive_trade_params(risk_budget)
    signals = build_signal_series(results_df, price_df)
    return simulate_backtest(price_df, signals, tp_pct, sl_pct, horizon, cost_bps, target_regime=target_regime)
