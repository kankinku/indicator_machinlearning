from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.backtest.tv_engine import TradingViewBacktestEngine, TVBacktestResult, TVTrade
from src.config import config
from src.shared.logger import get_logger

logger = get_logger("testing.tv_calibration")

SCENARIO_DEFAULTS: Dict[str, Dict[str, object]] = {
    "timing": {
        "commission_type": "percent",
        "commission_value": 0.0,
        "slippage_mode": "ticks",
        "slippage_value": 0.0,
    },
    "cost": {
        "commission_type": "percent",
        "commission_value": 0.1,
        "slippage_mode": "ticks",
        "slippage_value": 1.0,
    },
    "collision": {},
}


@dataclass
class CalibrationMismatch:
    code: str
    details: str
    likely_causes: List[str]


@dataclass
class CalibrationReport:
    scenario: str
    mode: str
    passed: bool
    mismatches: List[CalibrationMismatch]
    summary: Dict[str, float]


def load_price_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Calibration CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    date_col = None
    for col in ("date", "datetime", "timestamp"):
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        raise ValueError("Calibration CSV missing date/datetime/timestamp column")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            df[col] = df["close"]
    return df[["open", "high", "low", "close"]]


def build_sma_cross_signals(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> Tuple[pd.Series, pd.Series]:
    close = df["close"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    cross_up = (fast_ma.shift(1) <= slow_ma.shift(1)) & (fast_ma > slow_ma)
    cross_down = (fast_ma.shift(1) >= slow_ma.shift(1)) & (fast_ma < slow_ma)
    return cross_up.fillna(False), cross_down.fillna(False)


def build_rsi_reversion_signals(df: pd.DataFrame, length: int = 14, low: float = 30, high: float = 70) -> Tuple[pd.Series, pd.Series]:
    try:
        from ta.momentum import RSIIndicator
    except Exception as e:
        raise ImportError("ta is required for RSI calibration") from e

    rsi = RSIIndicator(close=df["close"], window=length).rsi()
    entry = (rsi < low).fillna(False)
    exit_sig = (rsi > high).fillna(False)
    return entry, exit_sig


def build_collision_signals(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    entry = pd.Series(True, index=df.index)
    exit_sig = pd.Series(False, index=df.index)
    return entry, exit_sig


def _serialize_trades(trades: List[TVTrade]) -> List[Dict[str, object]]:
    return [asdict(t) for t in trades]


def _format_summary(result: TVBacktestResult) -> Dict[str, float]:
    return {
        "final_equity": result.final_equity,
        "total_return_pct": result.total_return_pct,
        "mdd_pct": result.mdd_pct,
        "trade_count": len(result.trades),
        "total_fees": result.total_fees,
        "total_slippage": result.total_slippage,
    }


def _log_error(context: str, error: Exception) -> None:
    log_dir = Path(config.LOG_DIR) / "errors"
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "error_id": f"tv_calibration_{int(time.time())}",
        "timestamp": time.time(),
        "environment": "local",
        "context": context,
        "stack": str(error),
        "root_cause": "unknown",
        "resolution": "",
        "state": {},
    }
    target = log_dir / f"{payload['error_id']}.json"
    try:
        target.write_text(json.dumps(payload, indent=2))
    except Exception:
        logger.exception("Failed to write calibration error log")


def run_calibration(
    scenario: str,
    csv_path: Path,
    reference_path: Path,
    overrides: Optional[Dict[str, object]] = None,
) -> CalibrationReport:
    overrides = overrides or {}
    scenario_defaults = SCENARIO_DEFAULTS.get(scenario, {})
    merged_overrides = {**scenario_defaults, **overrides}
    try:
        if csv_path.exists():
            df = load_price_data(csv_path)
        else:
            df = _build_fallback_dataset(scenario)
        entry_sig, exit_sig, tp_pct, sl_pct, max_hold = _resolve_scenario(df, scenario, merged_overrides)

        engine = TradingViewBacktestEngine(
            commission_type=merged_overrides.get("commission_type", config.TV_CALIBRATION_COMMISSION_TYPE),
            commission_value=float(merged_overrides.get("commission_value", config.TV_CALIBRATION_COMMISSION_VALUE)),
            slippage_mode=merged_overrides.get("slippage_mode", config.TV_CALIBRATION_SLIPPAGE_MODE),
            slippage_value=float(merged_overrides.get("slippage_value", config.TV_CALIBRATION_SLIPPAGE_VALUE)),
            tick_size=float(merged_overrides.get("tick_size", config.TV_CALIBRATION_TICK_SIZE)),
            execution_timing=merged_overrides.get("execution_timing", config.TV_CALIBRATION_EXECUTION),
            tp_sl_priority=merged_overrides.get("tp_sl_priority", config.TV_CALIBRATION_TP_SL_PRIORITY),
            position_mode=merged_overrides.get("position_mode", config.TV_CALIBRATION_POSITION_MODE),
            position_value=float(merged_overrides.get("position_value", config.TV_CALIBRATION_POSITION_VALUE)),
            pyramiding=int(merged_overrides.get("pyramiding", config.TV_CALIBRATION_PYRAMIDING)),
            allow_same_bar_reentry=bool(merged_overrides.get("allow_same_bar_reentry", config.TV_CALIBRATION_ALLOW_SAME_BAR_REENTRY)),
        )

        result = engine.run(df, entry_sig, exit_sig, tp_pct=tp_pct, sl_pct=sl_pct, max_hold_bars=max_hold)
        engine_output = {
            "trades": _serialize_trades(result.trades),
            "summary": _format_summary(result),
        }

        if reference_path.exists():
            reference = _load_reference(reference_path, df)
            mismatches = compare_results(reference, engine_output, df)
            mode = "reference"
        else:
            mismatches = compare_with_invariants(
                scenario=scenario,
                df=df,
                entry_sig=entry_sig,
                exit_sig=exit_sig,
                result=result,
                overrides=merged_overrides,
            )
            mode = "fallback"

        passed = len(mismatches) == 0
        return CalibrationReport(
            scenario=scenario,
            mode=mode,
            passed=passed,
            mismatches=mismatches,
            summary=engine_output["summary"],
        )
    except Exception as e:
        _log_error(f"run_calibration:{scenario}", e)
        raise


def _resolve_scenario(
    df: pd.DataFrame,
    scenario: str,
    overrides: Dict[str, object],
) -> Tuple[pd.Series, pd.Series, Optional[float], Optional[float], Optional[int]]:
    if scenario == "timing":
        entry, exit_sig = build_sma_cross_signals(df)
        return entry, exit_sig, None, None, None
    if scenario == "cost":
        entry, exit_sig = build_rsi_reversion_signals(df)
        return entry, exit_sig, None, None, None
    if scenario == "collision":
        entry, exit_sig = build_collision_signals(df)
        tp_pct = float(overrides.get("tp_pct", config.TV_CALIBRATION_TP_PCT))
        sl_pct = float(overrides.get("sl_pct", config.TV_CALIBRATION_SL_PCT))
        max_hold = int(overrides.get("max_hold_bars", config.TV_CALIBRATION_MAX_HOLD_BARS))
        return entry, exit_sig, tp_pct, sl_pct, max_hold
    raise ValueError(f"Unknown calibration scenario: {scenario}")


def _build_fallback_dataset(scenario: str) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    if scenario == "collision":
        data = {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0] + [100.0] * 15,
            "high": [100.5, 101.5, 102.0, 100.5, 100.5] + [100.5] * 15,
            "low": [99.5, 98.5, 98.0, 99.5, 99.5] + [99.5] * 15,
            "close": [100.0, 100.0, 100.0, 100.0, 100.0] + [100.0] * 15,
        }
        return pd.DataFrame(data, index=dates)

    x = np.linspace(0, 4 * np.pi, len(dates))
    closes = 100.0 + (5.0 * np.sin(x))
    data = {
        "open": closes * 0.998,
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
    }
    return pd.DataFrame(data, index=dates)


def _load_reference(path: Path, df: pd.DataFrame) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Reference JSON not found: {path}")
    raw = json.loads(path.read_text())
    trades = raw.get("trades", [])
    summary = raw.get("summary", {})

    normalized = []
    for t in trades:
        entry_time = t.get("entry_time") or t.get("entry_date")
        exit_time = t.get("exit_time") or t.get("exit_date")
        normalized.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": float(t.get("entry_price", 0.0)),
            "exit_price": float(t.get("exit_price", 0.0)),
            "qty": float(t.get("qty", 0.0)),
            "pnl_net": float(t.get("pnl_net", t.get("return_pct", 0.0))),
            "reason": t.get("reason", ""),
            "entry_bar_idx": _resolve_bar_index(df, entry_time, t.get("entry_bar_idx")),
            "exit_bar_idx": _resolve_bar_index(df, exit_time, t.get("exit_bar_idx")),
        })

    return {"trades": normalized, "summary": summary}


def _resolve_bar_index(df: pd.DataFrame, time_value: Optional[str], fallback: Optional[int]) -> Optional[int]:
    if fallback is not None:
        try:
            return int(fallback)
        except (TypeError, ValueError):
            return None
    if not time_value:
        return None
    try:
        ts = pd.to_datetime(time_value)
        if ts in df.index:
            return int(df.index.get_loc(ts))
    except Exception:
        return None
    return None


def compare_results(reference: Dict[str, object], engine_output: Dict[str, object], df: pd.DataFrame) -> List[CalibrationMismatch]:
    mismatches: List[CalibrationMismatch] = []
    ref_trades = reference.get("trades", [])
    out_trades = engine_output.get("trades", [])

    if len(ref_trades) != len(out_trades):
        mismatches.append(_make_mismatch(
            "TRADE_COUNT_MISMATCH",
            f"ref={len(ref_trades)} out={len(out_trades)}",
        ))
        return mismatches

    price_tol = config.TV_CALIBRATION_TOL_PRICE_PCT
    for idx, (ref, out) in enumerate(zip(ref_trades, out_trades)):
        ref_entry_idx = ref.get("entry_bar_idx")
        ref_exit_idx = ref.get("exit_bar_idx")
        out_entry_idx = out.get("entry_bar_idx")
        out_exit_idx = out.get("exit_bar_idx")

        if ref_entry_idx is not None and out_entry_idx is not None and ref_entry_idx != out_entry_idx:
            mismatches.append(_make_mismatch(
                "ENTRY_BAR_MISMATCH",
                f"trade={idx} ref={ref_entry_idx} out={out_entry_idx}",
            ))
            break

        if ref_exit_idx is not None and out_exit_idx is not None and ref_exit_idx != out_exit_idx:
            mismatches.append(_make_mismatch(
                "EXIT_BAR_MISMATCH",
                f"trade={idx} ref={ref_exit_idx} out={out_exit_idx}",
            ))
            break

        if _pct_diff(ref.get("entry_price"), out.get("entry_price")) > price_tol:
            mismatches.append(_make_mismatch(
                "ENTRY_PRICE_MISMATCH",
                f"trade={idx} ref={ref.get('entry_price')} out={out.get('entry_price')}",
            ))
            break

        if _pct_diff(ref.get("exit_price"), out.get("exit_price")) > price_tol:
            mismatches.append(_make_mismatch(
                "EXIT_PRICE_MISMATCH",
                f"trade={idx} ref={ref.get('exit_price')} out={out.get('exit_price')}",
            ))
            break

        if ref.get("reason") and ref.get("reason") != out.get("reason"):
            mismatches.append(_make_mismatch(
                "EXIT_REASON_MISMATCH",
                f"trade={idx} ref={ref.get('reason')} out={out.get('reason')}",
            ))
            break

    ref_summary = reference.get("summary", {})
    out_summary = engine_output.get("summary", {})

    if "total_fees" in ref_summary:
        if _abs_diff(ref_summary.get("total_fees"), out_summary.get("total_fees")) > 1e-6:
            mismatches.append(_make_mismatch(
                "FEE_MISMATCH",
                f"ref={ref_summary.get('total_fees')} out={out_summary.get('total_fees')}",
            ))

    if "total_slippage" in ref_summary:
        if _abs_diff(ref_summary.get("total_slippage"), out_summary.get("total_slippage")) > 1e-6:
            mismatches.append(_make_mismatch(
                "SLIPPAGE_MISMATCH",
                f"ref={ref_summary.get('total_slippage')} out={out_summary.get('total_slippage')}",
            ))

    if "final_equity" in ref_summary:
        if _pct_diff(ref_summary.get("final_equity"), out_summary.get("final_equity")) > config.TV_CALIBRATION_TOL_FINAL_EQUITY_PCT:
            mismatches.append(_make_mismatch(
                "FINAL_EQUITY_MISMATCH",
                f"ref={ref_summary.get('final_equity')} out={out_summary.get('final_equity')}",
            ))

    if "mdd_pct" in ref_summary:
        if _abs_diff(ref_summary.get("mdd_pct"), out_summary.get("mdd_pct")) > (config.TV_CALIBRATION_TOL_MDD_PCT * 100.0):
            mismatches.append(_make_mismatch(
                "MDD_MISMATCH",
                f"ref={ref_summary.get('mdd_pct')} out={out_summary.get('mdd_pct')}",
            ))

    return mismatches


def compare_with_invariants(
    scenario: str,
    df: pd.DataFrame,
    entry_sig: pd.Series,
    exit_sig: pd.Series,
    result: TVBacktestResult,
    overrides: Dict[str, object],
) -> List[CalibrationMismatch]:
    mismatches: List[CalibrationMismatch] = []
    execution = overrides.get("execution_timing", config.TV_CALIBRATION_EXECUTION)
    price_tol = config.TV_CALIBRATION_TOL_PRICE_PCT

    trades = result.trades
    if scenario == "timing":
        if not trades:
            mismatches.append(_make_mismatch("TRADE_COUNT_MISMATCH", "no trades generated"))
            return mismatches

        for idx, trade in enumerate(trades):
            entry_idx = trade.entry_bar_idx
            if execution == "next_open":
                if entry_idx <= 0 or not entry_sig.iloc[entry_idx - 1]:
                    mismatches.append(_make_mismatch(
                        "ENTRY_BAR_MISMATCH",
                        f"trade={idx} entry_idx={entry_idx} signal_idx={entry_idx - 1}",
                    ))
                    break
                if _pct_diff(trade.entry_price, float(df["open"].iloc[entry_idx])) > price_tol:
                    mismatches.append(_make_mismatch(
                        "ENTRY_PRICE_MISMATCH",
                        f"trade={idx} entry={trade.entry_price} open={float(df['open'].iloc[entry_idx])}",
                    ))
                    break
            else:
                if not entry_sig.iloc[entry_idx]:
                    mismatches.append(_make_mismatch(
                        "ENTRY_BAR_MISMATCH",
                        f"trade={idx} entry_idx={entry_idx} signal_idx={entry_idx}",
                    ))
                    break
                if _pct_diff(trade.entry_price, float(df["close"].iloc[entry_idx])) > price_tol:
                    mismatches.append(_make_mismatch(
                        "ENTRY_PRICE_MISMATCH",
                        f"trade={idx} entry={trade.entry_price} close={float(df['close'].iloc[entry_idx])}",
                    ))
                    break

    if scenario == "cost":
        commission_type = overrides.get("commission_type", config.TV_CALIBRATION_COMMISSION_TYPE)
        commission_value = float(overrides.get("commission_value", config.TV_CALIBRATION_COMMISSION_VALUE))
        slippage_mode = overrides.get("slippage_mode", config.TV_CALIBRATION_SLIPPAGE_MODE)
        slippage_value = float(overrides.get("slippage_value", config.TV_CALIBRATION_SLIPPAGE_VALUE))
        tick_size = float(overrides.get("tick_size", config.TV_CALIBRATION_TICK_SIZE))

        expected_fees = 0.0
        expected_slip = 0.0
        for trade in trades:
            entry_fee = _calc_fee(commission_type, commission_value, trade.entry_price, trade.qty)
            exit_fee = _calc_fee(commission_type, commission_value, trade.exit_price, trade.qty)
            expected_fees += (entry_fee + exit_fee)

            slip_entry = _calc_slippage(slippage_mode, slippage_value, tick_size, trade.entry_price)
            slip_exit = _calc_slippage(slippage_mode, slippage_value, tick_size, trade.exit_price)
            expected_slip += (slip_entry + slip_exit) * trade.qty

            if abs(trade.pnl_net - (trade.pnl_gross - (entry_fee + exit_fee))) > 1e-6:
                mismatches.append(_make_mismatch(
                    "FEE_MISMATCH",
                    f"trade pnl_net mismatch trade={trade.pnl_net}",
                ))
                break

        if expected_fees and abs(expected_fees - result.total_fees) > 1e-6:
            mismatches.append(_make_mismatch(
                "FEE_MISMATCH",
                f"expected={expected_fees} actual={result.total_fees}",
            ))

        if expected_slip and abs(expected_slip - result.total_slippage) > 1e-6:
            mismatches.append(_make_mismatch(
                "SLIPPAGE_MISMATCH",
                f"expected={expected_slip} actual={result.total_slippage}",
            ))

    if scenario == "collision":
        if not trades:
            mismatches.append(_make_mismatch("TRADE_COUNT_MISMATCH", "no trades generated"))
            return mismatches
        priority = overrides.get("tp_sl_priority", config.TV_CALIBRATION_TP_SL_PRIORITY)
        expected_reason = "sl" if priority == "stop_first" else "tp"
        first = trades[0]
        if first.reason not in ("sl", "tp"):
            mismatches.append(_make_mismatch(
                "EXIT_REASON_MISMATCH",
                f"first_reason={first.reason}",
            ))
        elif first.reason != expected_reason:
            mismatches.append(_make_mismatch(
                "EXIT_REASON_MISMATCH",
                f"expected={expected_reason} actual={first.reason}",
            ))

    return mismatches


def _make_mismatch(code: str, details: str) -> CalibrationMismatch:
    causes = config.TV_CALIBRATION_CAUSE_MAP.get(code, [])
    return CalibrationMismatch(code=code, details=details, likely_causes=causes[:3])


def _pct_diff(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return 0.0
    if b == 0:
        return abs(a - b)
    return abs(a - b) / abs(b)


def _calc_fee(commission_type: str, commission_value: float, price: float, qty: float) -> float:
    if commission_type == "percent":
        return price * qty * (commission_value / 100.0)
    if commission_type == "bps":
        return price * qty * (commission_value / 10000.0)
    if commission_type == "fixed":
        return commission_value
    return 0.0


def _calc_slippage(slippage_mode: str, slippage_value: float, tick_size: float, price: float) -> float:
    if slippage_mode == "ticks":
        return slippage_value * tick_size
    if slippage_mode == "bps":
        return price * (slippage_value / 10000.0)
    if slippage_mode == "percent":
        return price * (slippage_value / 100.0)
    return 0.0


def _abs_diff(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return 0.0
    return abs(a - b)
