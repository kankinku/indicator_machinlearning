from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from src.shared.logger import get_logger

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
    complexity_penalty: float = 0.0
    trade_returns: List[float] = None

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
    
    def run(self, close_prices: pd.Series, entry_signals: pd.Series, exit_signals: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> BacktestResult:
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
        
        # State Machine Loop
        for i in range(len(prices)):
            # 1. State: LONG -> Check EXIT
            if state == 1:
                if exits.iloc[i]:
                    self._record_trade(trades, prices, entry_idx, i, dates)
                    state = 0
                    entry_idx = None
            
            # 2. State: FLAT -> Check ENTRY
            elif state == 0:
                if entries.iloc[i]:
                    if not exits.iloc[i]:
                        state = 1
                        entry_idx = i
            
            positions[i] = state
            
        if state == 1:
            self._record_trade(trades, prices, entry_idx, len(prices)-1, dates)
            
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
        excess_return = total_return - bh_return
        
        avg_trade_return = float(np.mean(trade_returns) * 100.0) if trade_count > 0 else 0.0
        
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
            trade_returns=trade_returns.tolist()
        )

    def _record_trade(self, trades, prices, entry_idx, exit_idx, dates):
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
            "bars": int(exit_idx - entry_idx)
        })

    def _empty_result(self) -> BacktestResult:
        return BacktestResult([], [], [], 0.0, 0.0, 0, 0.0, 1.0, 0.0, 0.0, trade_returns=[])
