from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from src.shared.logger import get_logger
from src.config import config

logger = get_logger("backtest.engine")

@dataclass
class BacktestResult:
    dates: List[str]
    equity_curve: List[float]
    trades: List[Dict[str, object]]
    total_return: float
    mdd: float
    trade_count: int
    win_rate: float
    profit_factor: float
    exposure: float
    avg_trade_return: float
    trades_per_year: float = 0.0
    excess_return: float = 0.0
    benchmark_return: float = 0.0
    complexity_penalty: float = 0.0
    trade_returns: List[float] = None
    invalid_action_count: int = 0
    invalid_action_rate: float = 0.0
    ignored_action_count: int = 0
    ignored_action_rate: float = 0.0
    total_action_bars: int = 0
    invalid_action_events: Optional[List[Dict[str, object]]] = None
    invalid_action_reason_counts: Optional[Dict[str, int]] = None
    invalid_action_first_index: Optional[int] = None
    ignored_action_events: Optional[List[Dict[str, object]]] = None
    ignored_action_reason_counts: Optional[Dict[str, int]] = None
    ignored_action_first_index: Optional[int] = None

class DeterministicBacktestEngine:
    """
    V13-PRO Engine: High-Precision State Machine.
    
    Rules:
    1. Decision at Close[i] -> Execution at Close[i].
    2. Profit applies to the interval (i, i+1).
    3. Simultaneous Signal: EXIT priority (Exit first, then check entry for NEXT bar).
    4. No same-bar re-entry: If exited at Close[i], cannot re-enter at the same Close[i].
    """
    def __init__(self, commission_bps: float = 0.0):
        self.commission_bps = commission_bps
        self.commission_frac = commission_bps / 10000.0
    
    def run(
        self,
        close_prices: pd.Series,
        entry_signals: pd.Series,
        exit_signals: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        tp_pct: Optional[float] = None,
        sl_pct: Optional[float] = None,
        max_hold_bars: Optional[int] = None,
        hold_min_bars: Optional[pd.Series] = None,
        hold_max_bars: Optional[pd.Series] = None,
        benchmark_mode: Optional[str] = None,
        benchmark_return_pct: Optional[float] = None,
    ) -> BacktestResult:
        if close_prices.empty:
            return self._empty_result()
            
        # Align signals
        entries = entry_signals.reindex(close_prices.index).fillna(False).astype(bool)
        exits = exit_signals.reindex(close_prices.index).fillna(False).astype(bool)
        
        dates = close_prices.index
        prices = close_prices.values
        
        positions = np.zeros(len(prices), dtype=int)
        trades = []
        
        state = 0 # 0: FLAT, 1: LONG
        entry_idx = None
        entry_price = None
        hold_bars = 0
        current_min_hold = 0
        current_max_hold = max_hold_bars
        invalid_action_count = 0
        ignored_action_count = 0
        total_action_bars = 0
        invalid_action_events: List[Dict[str, object]] = []
        invalid_action_reason_counts: Dict[str, int] = {}
        invalid_action_first_index: Optional[int] = None
        ignored_action_events: List[Dict[str, object]] = []
        ignored_action_reason_counts: Dict[str, int] = {}
        ignored_action_first_index: Optional[int] = None
        max_invalid_events = int(getattr(config, "INVALID_ACTION_MAX_EVENTS", 5))

        def _record_invalid(reason: str, state_value: int, idx: int, entry: bool, exit: bool) -> None:
            nonlocal invalid_action_count, invalid_action_first_index
            invalid_action_count += 1
            invalid_action_reason_counts[reason] = invalid_action_reason_counts.get(reason, 0) + 1
            if invalid_action_first_index is None:
                invalid_action_first_index = int(idx)
            if len(invalid_action_events) >= max_invalid_events:
                return
            position_state = "LONG" if state_value == 1 else "FLAT"
            allowed = ["HOLD", "EXIT_LONG"] if state_value == 1 else ["HOLD", "ENTER_LONG"]
            action = "CONFLICT_ENTRY_EXIT"
            if reason == "FLAT_EXIT":
                action = "EXIT_LONG"
            elif reason == "LONG_ENTER":
                action = "ENTER_LONG"
            invalid_action_events.append({
                "step_index": int(idx),
                "position_state_before": position_state,
                "action_chosen": action,
                "allowed_actions_in_state": allowed,
                "invalid_reason_code": reason,
                "entry_signal": bool(entry),
                "exit_signal": bool(exit),
            })

        def _record_ignored(reason: str, state_value: int, idx: int, entry: bool, exit: bool) -> None:
            nonlocal ignored_action_count, ignored_action_first_index
            ignored_action_count += 1
            ignored_action_reason_counts[reason] = ignored_action_reason_counts.get(reason, 0) + 1
            if ignored_action_first_index is None:
                ignored_action_first_index = int(idx)
            if len(ignored_action_events) >= max_invalid_events:
                return
            position_state = "LONG" if state_value == 1 else "FLAT"
            allowed = ["HOLD", "EXIT_LONG"] if state_value == 1 else ["HOLD", "ENTER_LONG"]
            action = "CONFLICT_ENTRY_EXIT"
            if reason == "IGNORED_FLAT_EXIT":
                action = "EXIT_LONG"
            elif reason == "IGNORED_LONG_ENTER":
                action = "ENTER_LONG"
            ignored_action_events.append({
                "step_index": int(idx),
                "position_state_before": position_state,
                "action_chosen": action,
                "allowed_actions_in_state": allowed,
                "ignored_reason_code": reason,
                "entry_signal": bool(entry),
                "exit_signal": bool(exit),
            })
        
        # State Machine Loop
        # Action set (long-only): HOLD / ENTER_LONG / EXIT_LONG
        for i in range(len(prices)):
            raw_entry = bool(entries.iloc[i])
            raw_exit = bool(exits.iloc[i])

            if raw_entry or raw_exit:
                total_action_bars += 1

            conflict = raw_entry and raw_exit
            invalid_reason = None
            ignored_reason = None
            if conflict:
                invalid_reason = "CONFLICT_ENTRY_EXIT"
            elif raw_entry and state == 1:
                ignored_reason = "IGNORED_LONG_ENTER"
            elif raw_exit and state == 0:
                ignored_reason = "IGNORED_FLAT_EXIT"
            if invalid_reason:
                _record_invalid(invalid_reason, state, i, raw_entry, raw_exit)
            if ignored_reason:
                _record_ignored(ignored_reason, state, i, raw_entry, raw_exit)

            # State-gated actions (signals themselves are not actions).
            entry_action = raw_entry and state == 0 and not conflict
            exit_action = raw_exit and state == 1 and not conflict

            # 1. State: LONG -> Check EXIT
            if state == 1:
                if exit_action and hold_bars >= int(current_min_hold or 0):
                    self._record_trade(trades, prices, entry_idx, i, dates, "AGENT_EXIT")
                    state = 0
                    entry_idx = None
                    entry_price = None
                    hold_bars = 0
                    current_min_hold = 0
                    current_max_hold = max_hold_bars
                else:
                    hold_bars += 1
                    if entry_price is not None:
                        if tp_pct is not None and prices[i] >= entry_price * (1.0 + tp_pct):
                            self._record_trade(trades, prices, entry_idx, i, dates, "TP")
                            state = 0
                            entry_idx = None
                            entry_price = None
                            hold_bars = 0
                            current_min_hold = 0
                            current_max_hold = max_hold_bars
                        elif sl_pct is not None and prices[i] <= entry_price * (1.0 - sl_pct):
                            self._record_trade(trades, prices, entry_idx, i, dates, "SL")
                            state = 0
                            entry_idx = None
                            entry_price = None
                            hold_bars = 0
                            current_min_hold = 0
                            current_max_hold = max_hold_bars
                        elif current_max_hold is not None and hold_bars >= int(current_max_hold):
                            self._record_trade(trades, prices, entry_idx, i, dates, "HORIZON")
                            state = 0
                            entry_idx = None
                            entry_price = None
                            hold_bars = 0
                            current_min_hold = 0
                            current_max_hold = max_hold_bars
            
            # 2. State: FLAT -> Check ENTRY
            elif state == 0:
                if entry_action and not exit_action:
                    state = 1
                    entry_idx = i
                    entry_price = prices[i]
                    hold_bars = 0
                    current_min_hold = 0
                    current_max_hold = max_hold_bars
                    if hold_min_bars is not None:
                        try:
                            current_min_hold = int(hold_min_bars.iloc[i])
                        except Exception:
                            current_min_hold = 0
                    if hold_max_bars is not None:
                        try:
                            current_max_hold = int(hold_max_bars.iloc[i])
                        except Exception:
                            current_max_hold = max_hold_bars
                    if current_min_hold < 0:
                        current_min_hold = 0
                    if current_max_hold is not None and current_max_hold < current_min_hold:
                        current_max_hold = current_min_hold
            
            positions[i] = state
            
        if state == 1:
            self._record_trade(trades, prices, entry_idx, len(prices)-1, dates, "HORIZON")
            
        # [V13.5] Logic: Signal[i] -> Applied to returns[i+1]
        price_returns = np.diff(prices) / prices[:-1]
        strategy_returns = positions[:-1] * price_returns
        
        # Apply commission on entries and exits
        # This is a bit tricky for equity curve. 
        # A simpler way is to subtract commission from trade returns and rebuild equity.
        trade_returns = np.array([t['return_pct'] / 100.0 for t in trades])
        
        if len(trades) > 0:
            trade_prod = np.prod(1.0 + trade_returns)
            cum_returns_final = trade_prod
        else:
            cum_returns_final = 1.0
            
        # Re-calculating equity curve correctly with commissions is complex if we want it bar-by-bar.
        # For now, we'll use the simple positions * returns and adjust final return if needed?
        # Actually, let's keep the engine's internal equity curve as raw, 
        # but the metrics will use the trade-based (commission-adjusted) returns.
        
        cum_returns = np.concatenate(([1.0], np.cumprod(1.0 + strategy_returns)))
        equity_curve = (cum_returns - 1.0) * 100.0
        total_return = (cum_returns_final - 1.0) * 100.0
        
        # Metrics
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns / peak) - 1.0
        mdd = abs(float(drawdown.min())) * 100.0
        
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns <= 0]
        
        trade_count = len(trades)
        win_rate = float(np.sum(trade_returns > 0) / trade_count) if trade_count > 0 else 0.0
        
        gross_profit = float(np.sum(wins))
        gross_loss = float(abs(np.sum(losses)))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (100.0 if gross_profit > 0 else 1.0)
        
        years = len(prices) / 252.0 if len(prices) > 0 else 1.0
        trades_per_year = trade_count / years
        exposure = float(np.mean(positions))
        
        bh_return = (prices[-1]/prices[0] - 1.0) * 100.0
        if benchmark_mode:
            from src.shared.benchmark import compute_benchmark_return, compute_excess_return
            bench_ret = compute_benchmark_return(
                prices=close_prices,
                exposure_ratio=exposure,
                mode=benchmark_mode,
                fixed_return_pct=float(benchmark_return_pct or 0.0),
            )
            excess_return = compute_excess_return(total_return, bench_ret)
        else:
            bench_ret = bh_return
            excess_return = total_return - bh_return
        
        avg_trade_return = float(np.mean(trade_returns) * 100.0) if trade_count > 0 else 0.0
        
        invalid_action_rate = 0.0
        ignored_action_rate = 0.0
        if total_action_bars > 0:
            invalid_action_rate = float(invalid_action_count / total_action_bars)
            ignored_action_rate = float(ignored_action_count / total_action_bars)

        return BacktestResult(
            dates=[d.strftime("%Y-%m-%d") for d in dates],
            equity_curve=equity_curve.tolist(),
            trades=trades,
            total_return=total_return,
            mdd=mdd,
            trade_count=trade_count,
            win_rate=win_rate,
            profit_factor=profit_factor,
            exposure=exposure,
            avg_trade_return=avg_trade_return,
            trades_per_year=trades_per_year,
            excess_return=excess_return,
            benchmark_return=bench_ret,
            trade_returns=trade_returns.tolist(),
            invalid_action_count=invalid_action_count,
            invalid_action_rate=invalid_action_rate,
            ignored_action_count=ignored_action_count,
            ignored_action_rate=ignored_action_rate,
            total_action_bars=total_action_bars,
            invalid_action_events=invalid_action_events,
            invalid_action_reason_counts=invalid_action_reason_counts,
            invalid_action_first_index=invalid_action_first_index,
            ignored_action_events=ignored_action_events,
            ignored_action_reason_counts=ignored_action_reason_counts,
            ignored_action_first_index=ignored_action_first_index,
        )

    def _record_trade(self, trades, prices, entry_idx, exit_idx, dates, exit_reason: str):
        raw_entry_price = prices[entry_idx]
        raw_exit_price = prices[exit_idx]
        
        # Effective prices with commission
        entry_price = raw_entry_price * (1.0 + self.commission_frac)
        exit_price = raw_exit_price * (1.0 - self.commission_frac)
        
        ret_pct = (exit_price / entry_price - 1.0) * 100.0
        
        trades.append({
            "entry_date": str(dates[entry_idx].date()),
            "exit_date": str(dates[exit_idx].date()),
            "entry_idx": int(entry_idx),
            "exit_idx": int(exit_idx),
            "entry_price": float(raw_entry_price),
            "exit_price": float(raw_exit_price),
            "net_entry_price": float(entry_price),
            "net_exit_price": float(exit_price),
            "return_pct": float(ret_pct),
            "exit_reason": exit_reason,
            "bars": int(exit_idx - entry_idx)
        })

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            [],
            [],
            [],
            0.0,
            0.0,
            0,
            0.0,
            1.0,
            0.0,
            0.0,
            trade_returns=[],
            invalid_action_count=0,
            invalid_action_rate=0.0,
            ignored_action_count=0,
            ignored_action_rate=0.0,
            total_action_bars=0,
            invalid_action_events=[],
            invalid_action_reason_counts={},
            invalid_action_first_index=None,
            ignored_action_events=[],
            ignored_action_reason_counts={},
            ignored_action_first_index=None,
        )
