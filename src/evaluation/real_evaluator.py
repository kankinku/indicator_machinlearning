from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from src.backtest.engine import DeterministicBacktestEngine
from src.config import config
from src.l1_judge.evaluator import SampleMetrics, compute_sample_metrics
from src.shared.backtest import derive_trade_params
from src.shared.observability import compute_cycle_stats

class RealEvaluator:
    """
    [V14] Real Evaluator
    Executes a high-fidelity deterministic backtest and computes comprehensive metrics.
    """
    def __init__(self, cost_bps: float = 5.0):
        self.engine = DeterministicBacktestEngine(commission_bps=cost_bps)
        self.cost_bps = cost_bps

    def evaluate(
        self,
        df: pd.DataFrame,
        entry_signals: pd.Series,
        exit_signals: pd.Series,
        risk_budget: Dict[str, Any],
        target_regime: Optional[str] = None,
        complexity_score: float = 0.0,
        act_signals: Optional[pd.Series] = None,
        hold_min_bars: Optional[pd.Series] = None,
        hold_max_bars: Optional[pd.Series] = None
    ) -> Tuple[SampleMetrics, Any]:
        """
        Runs the backtest and computes unified metrics.
        """
        tp_pct, sl_pct, horizon = derive_trade_params(risk_budget)
        bt_result = self.engine.run(
            df["close"],
            entry_signals,
            exit_signals,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            max_hold_bars=horizon,
            hold_min_bars=hold_min_bars,
            hold_max_bars=hold_max_bars,
            benchmark_mode=getattr(config, "EVAL_BENCHMARK_MODE", "bh"),
            benchmark_return_pct=float(getattr(config, "EVAL_BENCHMARK_RETURN_PCT", 0.0)),
        )
        
        trade_returns = np.array(bt_result.trade_returns)
        
        # Build full strategy returns series for EquityStats
        equity = np.array(bt_result.equity_curve) / 100.0 + 1.0
        full_returns = np.diff(equity) / equity[:-1]
        
        bench_ret = float(getattr(bt_result, "benchmark_return", 0.0))
        metrics = compute_sample_metrics(
            trade_returns=trade_returns,
            trade_count=bt_result.trade_count,
            bars_total=len(df),
            benchmark_roi_pct=bench_ret,
            full_returns=full_returns,
            exposure_mask=None,
            complexity_score=complexity_score
        )
        
        # Override exposure ratio from engine (accurate mean).
        metrics.equity.exposure_ratio = float(bt_result.exposure)
        metrics.equity.percent_in_market = float(bt_result.exposure)
        metrics.equity.total_return_pct = float(bt_result.total_return)
        metrics.equity.benchmark_roi_pct = float(bench_ret)
        metrics.equity.excess_return = float(bt_result.excess_return)

        # Signal degeneracy metrics.
        if entry_signals is None:
            entry_count = 0
        else:
            entry_count = int(entry_signals.astype(bool).sum())
        metrics.trades.entry_signal_rate = float(entry_count / max(1, len(df)))
        if bt_result.trades:
            metrics.trades.avg_holding_bars = float(np.mean([t.get("bars", 0) for t in bt_result.trades]))
        else:
            metrics.trades.avg_holding_bars = 0.0

        cycles, stats = compute_cycle_stats(bt_result.trades, len(df))
        metrics.trades.cycle_count = int(stats["cycle_count"])
        metrics.trades.entry_count = int(stats["entry_count"])
        metrics.trades.exit_count = int(stats["exit_count"])
        metrics.trades.invalid_action_count = int(getattr(bt_result, "invalid_action_count", 0))
        metrics.trades.invalid_action_rate = float(getattr(bt_result, "invalid_action_rate", 0.0))
        metrics.trades.ignored_action_count = int(getattr(bt_result, "ignored_action_count", 0))
        metrics.trades.ignored_action_rate = float(getattr(bt_result, "ignored_action_rate", 0.0))
        metrics.trades.cost_enter = int(stats["entry_count"])
        metrics.trades.cost_flip = int(stats["entry_count"] + stats["exit_count"])
        if act_signals is not None:
            act_count = int(act_signals.astype(bool).sum())
        else:
            act_count = int((entry_signals.astype(bool) | exit_signals.astype(bool)).sum())
        metrics.trades.act_count = act_count
        metrics.trades.act_rate = float(act_count / max(1, len(df)))
        
        return metrics, bt_result

    def get_alpha(self, metrics: SampleMetrics) -> float:
        """
        Calculates Alpha using unified excess_return field.
        """
        return metrics.equity.excess_return
