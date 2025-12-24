from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from src.backtest.engine import DeterministicBacktestEngine
from src.l1_judge.evaluator import SampleMetrics, compute_sample_metrics

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
        signals: pd.Series,
        risk_budget: Dict[str, Any],
        target_regime: Optional[str] = None
    ) -> Tuple[SampleMetrics, Any]:
        """
        Runs the backtest and computes unified metrics.
        """
        exit_sig = pd.Series(0, index=signals.index)
        bt_result = self.engine.run(df["close"], signals, exit_sig)
        
        trade_returns = np.array(bt_result.trade_returns)
        
        # Build full strategy returns series for EquityStats
        # Since cumulative equity = (1+r1)(1+r2)... we can work back to daily returns r
        # Or better: use the equity_curve from engine (already contains strategy returns)
        equity = np.array(bt_result.equity_curve) / 100.0 + 1.0
        full_returns = np.diff(equity) / equity[:-1]
        
        # Benchmark ROI
        bench_ret = (df["close"].iloc[-1] / df["close"].iloc[0] - 1.0) * 100.0 if not df.empty else 0.0
        
        metrics = compute_sample_metrics(
            trade_returns=trade_returns,
            trade_count=bt_result.trade_count,
            bars_total=len(df),
            benchmark_roi_pct=bench_ret,
            full_returns=full_returns,
            exposure_mask=None # bt_result.exposure is already a mean, we pass it indirectly if needed
        )
        
        # Override exposure_ratio from engine (accurate mean)
        # Note: metrics (WindowMetrics) has .equity.exposure_ratio
        # Dataclasses are frozen, so we would need a new instance or use compute_sample_metrics argument
        # Re-running with exposure_mask=None but we can set it if we had the positions series.
        # DeterministicBacktestEngine doesn't return the full positions series in BacktestResult currently.
        # We can just trust bt_result.exposure for now.
        
        return metrics, bt_result

    def get_alpha(self, metrics: SampleMetrics) -> float:
        """
        Calculates Alpha using unified excess_return field.
        """
        return metrics.equity.excess_return
