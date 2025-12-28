from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from src.config import config

@dataclass
class TradeStats:
    """
    Statistics for individual trades.
    """
    trade_count: int = 0
    valid_trade_count: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 1.0
    expectancy: float = 0.0
    reward_risk: float = 0.0  # Median win / |Median loss|
    top1_share: float = 0.0
    top3_share: float = 0.0
    trades_per_year: float = 0.0
    entry_signal_rate: float = 0.0
    avg_holding_bars: float = 0.0
    cycle_count: int = 0
    entry_count: int = 0
    exit_count: int = 0
    invalid_action_count: int = 0
    invalid_action_rate: float = 0.0

@dataclass
class EquityStats:
    """
    Statistics for the equity curve / returns series.
    """
    total_return_pct: float = 0.0
    cagr: float = 0.0
    vol_pct: float = 0.0  # Annualized volatility
    sharpe: float = 0.0   # Annualized Sharpe ratio
    max_drawdown_pct: float = 0.0
    mdd_duration: int = 0  # Max drawdown duration in bars
    exposure_ratio: float = 0.0
    percent_in_market: float = 0.0
    benchmark_roi_pct: float = 0.0
    excess_return: float = 0.0

@dataclass
class WindowMetrics:
    """
    Aggregated metrics for a single evaluation window.
    """
    window_id: str
    trades: TradeStats
    equity: EquityStats
    is_rejected: bool = False
    rejection_reason: str = "PASS"
    raw_score: float = 0.0
    bars_total: int = 0
    complexity_score: float = 0.0 # [V17] Structural AST complexity
    failure_codes: List[str] = field(default_factory=list)
    
    # Metadata
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AggregateMetrics:
    """
    Final aggregated metrics across multiple windows (SSOT).
    """
    # Summary of individual windows
    window_results: List[WindowMetrics] = field(default_factory=list)
    
    # Aggregated Stats (Mean/Median)
    mean_return: float = 0.0
    median_return: float = 0.0
    mean_sharpe: float = 0.0
    mean_mdd: float = 0.0
    
    # Stability & Risk
    violation_rate: float = 0.0
    score_p10: float = 0.0  # Pessimistic score (10th percentile)
    score_std: float = 0.0  # Score variance
    
    # Final Decision
    final_score: float = 0.0
    mean_complexity: float = 0.0 # [V17]
    
    # Artifacts (captured from the best window/sample)
    best_sample_id: Optional[str] = None
    reward_breakdown: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# =============================================================================
# Core Calculation Logic (Pure Functions)
# =============================================================================

def compute_trades_stats(trade_returns: np.ndarray, bars_total: int) -> TradeStats:
    """
    Calculates statistics based on a series of trade returns.
    """
    if trade_returns.size == 0:
        return TradeStats()

    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    
    tc = len(trade_returns)
    win_count = len(wins)
    win_rate = win_count / tc if tc > 0 else 0.0
    
    avg_win = float(np.mean(wins)) if wins.size > 0 else 0.0
    avg_loss = float(np.mean(losses)) if losses.size > 0 else 0.0
    
    # Profit Factor
    gross_profit = np.sum(wins) if wins.size > 0 else 0.0
    gross_loss = abs(np.sum(losses)) if losses.size > 0 else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 1e-9 else (100.0 if gross_profit > 0 else 1.0)
    
    # Expectancy: (WinRate * AvgWin) - (LossRate * |AvgLoss|)
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
    
    # Reward Risk (Median based for robustness)
    if wins.size > 0 and losses.size > 0:
        rr = np.median(wins) / abs(np.median(losses))
    else:
        rr = 0.0
        
    # Anti-Luck: Share of top trades
    total_pos_pnl = np.sum(wins) if wins.size > 0 else 0.0
    if total_pos_pnl > 1e-9:
        sorted_pos = np.sort(wins)[::-1]
        top1_share = float(sorted_pos[0] / total_pos_pnl)
        top3_share = float(np.sum(sorted_pos[:3]) / total_pos_pnl)
    else:
        top1_share = 0.0
        top3_share = 0.0
        
    # Annualized Trades
    years = bars_total / 252.0 if bars_total > 0 else 1.0
    trades_per_year = tc / years if years > 0 else 0.0
    
    return TradeStats(
        trade_count=tc,
        valid_trade_count=tc, # Placeholder unless filtered externally
        win_rate=float(win_rate),
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        expectancy=float(expectancy),
        reward_risk=float(rr),
        top1_share=top1_share,
        top3_share=top3_share,
        trades_per_year=float(trades_per_year),
        cycle_count=int(tc),
        entry_count=int(tc),
        exit_count=int(tc),
    )

def compute_equity_stats(
    returns: np.ndarray, 
    bars_total: int, 
    benchmark_roi_pct: float = 0.0,
    exposure_mask: Optional[np.ndarray] = None
) -> EquityStats:
    """
    Calculates equity-based statistics.
    Returns should be a 1D array of percentage returns (0.01 = 1%).
    """
    if returns.size == 0:
        return EquityStats(benchmark_roi_pct=benchmark_roi_pct, excess_return=-benchmark_roi_pct)

    # Convert to log returns for compounding if necessary, but standard is cumprod
    equity = np.cumprod(1.0 + returns)
    total_return_pct = float((equity[-1] - 1.0) * 100.0)
    
    # CAGR
    years = bars_total / 252.0 if bars_total > 0 else 1.0
    if equity[-1] > 0 and years > 0:
        cagr = float(((equity[-1]) ** (1.0 / years) - 1.0) * 100.0)
    else:
        cagr = -100.0
        
    # MDD
    peak = np.maximum.accumulate(equity)
    drawdown = (equity / peak) - 1.0
    max_mdd_pct = float(abs(np.min(drawdown)) * 100.0)
    
    # MDD Duration
    is_in_dd = drawdown < 0
    if not np.any(is_in_dd):
        mdd_duration = 0
    else:
        # Simple max consecutive True in is_in_dd
        groups = np.split(is_in_dd, np.where(np.diff(is_in_dd.astype(int)) != 0)[0] + 1)
        mdd_duration = int(max([len(g) for g in groups if g[0]]) if any(g[0] for g in groups) else 0)
        
    # Volatility (Annualized)
    daily_vol = np.std(returns)
    vol_pct = float(daily_vol * np.sqrt(252) * 100.0)
    
    # Sharpe (Annualized, assuming 0% risk free)
    sharpe = float(cagr / vol_pct) if vol_pct > 1e-9 else 0.0
    
    # Exposure
    if exposure_mask is not None:
        exposure_ratio = float(np.mean(exposure_mask))
    else:
        exposure_ratio = 0.0
    percent_in_market = exposure_ratio
        
    excess_return = total_return_pct - benchmark_roi_pct
    
    return EquityStats(
        total_return_pct=total_return_pct,
        cagr=cagr,
        vol_pct=vol_pct,
        sharpe=sharpe,
        max_drawdown_pct=max_mdd_pct,
        mdd_duration=mdd_duration,
        exposure_ratio=exposure_ratio,
        percent_in_market=percent_in_market,
        benchmark_roi_pct=benchmark_roi_pct,
        excess_return=float(excess_return)
    )

def compute_window_metrics(
    window_id: str,
    trade_returns: np.ndarray,
    full_returns: np.ndarray,
    bars_total: int,
    benchmark_roi_pct: float = 0.0,
    exposure_mask: Optional[np.ndarray] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    complexity_score: float = 0.0
) -> WindowMetrics:
    """
    Combined helper to get both trade and equity stats for a window.
    """
    t_stats = compute_trades_stats(trade_returns, bars_total)
    e_stats = compute_equity_stats(full_returns, bars_total, benchmark_roi_pct, exposure_mask)
    
    return WindowMetrics(
        window_id=window_id,
        trades=t_stats,
        equity=e_stats,
        bars_total=bars_total,
        start_date=start_date,
        end_date=end_date,
        complexity_score=complexity_score
    )

# =============================================================================
# Aggregation & Scoring
# =============================================================================

def aggregate_windows(
    window_results: List[WindowMetrics],
    eval_score_override: Optional[float] = None
) -> AggregateMetrics:
    """
    Aggregates metrics from multiple evaluation windows.
    """
    if not window_results:
        return AggregateMetrics()
        
    returns = [w.equity.total_return_pct for w in window_results]
    sharpes = [w.equity.sharpe for w in window_results]
    mdds = [w.equity.max_drawdown_pct for w in window_results]
    violations = [w.is_rejected for w in window_results]
    
    mean_ret = float(np.mean(returns))
    median_ret = float(np.median(returns))
    mean_sharpe = float(np.mean(sharpes))
    mean_mdd = float(np.mean(mdds))
    mean_complexity = float(np.mean([w.complexity_score for w in window_results]))
    
    violation_rate = float(sum(violations) / len(violations))
    
    # Distribution of returns for pessimistic scoring
    score_p10 = float(np.quantile(returns, 0.1))
    score_std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    
    # Fallback to mean_ret if no override
    final_score = eval_score_override if eval_score_override is not None else mean_ret
    
    return AggregateMetrics(
        window_results=window_results,
        mean_return=mean_ret,
        median_return=median_ret,
        mean_sharpe=mean_sharpe,
        mean_mdd=mean_mdd,
        violation_rate=violation_rate,
        score_p10=score_p10,
        score_std=score_std,
        final_score=final_score,
        mean_complexity=mean_complexity
    )

# =============================================================================
# Legacy Compatibility Layer
# =============================================================================

def metrics_to_legacy_dict(agg: AggregateMetrics) -> Dict[str, Any]:
    """
    Maps the new AggregateMetrics to the legacy dictionary format 
    expected by dashboard and infinite_loop.
    """
    # Usually we want the aggregated view
    total_trades = int(np.sum([w.trades.trade_count for w in agg.window_results])) if agg.window_results else 0
    total_bars = sum([w.bars_total for w in agg.window_results]) if agg.window_results else 252
    entry_rate = total_trades / max(1, total_bars)
    res = {
        "total_return_pct": agg.mean_return,
        "mdd_pct": agg.mean_mdd,
        "win_rate": np.mean([w.trades.win_rate for w in agg.window_results]) if agg.window_results else 0.0,
        "n_trades": total_trades,
        "valid_trade_count": int(np.sum([w.trades.valid_trade_count for w in agg.window_results])) if agg.window_results else 0,
        "reward_risk": np.mean([w.trades.reward_risk for w in agg.window_results]) if agg.window_results else 0.0,
        "sharpe": agg.mean_sharpe,
        "trades_per_year": np.mean([w.trades.trades_per_year for w in agg.window_results]) if agg.window_results else 0.0,
        "profit_factor": np.mean([w.trades.profit_factor for w in agg.window_results]) if agg.window_results else 1.0,
        "violation_rate": agg.violation_rate,
        "final_score": agg.final_score,
        "top1_share": np.mean([w.trades.top1_share for w in agg.window_results]) if agg.window_results else 0.0,
        "top3_share": np.mean([w.trades.top3_share for w in agg.window_results]) if agg.window_results else 0.0,
        "excess_return": np.mean([w.equity.excess_return for w in agg.window_results]) if agg.window_results else 0.0,
        "complexity_score": agg.mean_complexity,
        "entry_signal_rate": entry_rate,
        "avg_holding_bars": np.mean([w.trades.avg_holding_bars for w in agg.window_results]) if agg.window_results else 0.0,
        "percent_in_market": np.mean([w.equity.percent_in_market for w in agg.window_results]) if agg.window_results else 0.0,
        "cycle_count": int(np.sum([w.trades.cycle_count for w in agg.window_results])) if agg.window_results else 0,
        "invalid_action_count": int(np.sum([w.trades.invalid_action_count for w in agg.window_results])) if agg.window_results else 0,
        "invalid_action_rate": np.mean([w.trades.invalid_action_rate for w in agg.window_results]) if agg.window_results else 0.0,
    }
    
    # Add oos_bars (sum of all windows)
    res["oos_bars"] = total_bars
    return res
