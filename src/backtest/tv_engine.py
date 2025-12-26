from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.shared.logger import get_logger

logger = get_logger("backtest.tv_engine")


@dataclass
class TVTrade:
    entry_time: str
    exit_time: str
    entry_bar_idx: int
    exit_bar_idx: int
    entry_price: float
    exit_price: float
    qty: float
    pnl_gross: float
    pnl_net: float
    fees: float
    slippage: float
    reason: str


@dataclass
class TVBacktestResult:
    trades: List[TVTrade]
    equity_curve: List[float]
    final_equity: float
    total_return_pct: float
    mdd_pct: float
    total_fees: float
    total_slippage: float


class TradingViewBacktestEngine:
    """
    TradingView-aligned backtest engine with explicit execution and cost rules.
    Long-only by default for calibration.
    """
    def __init__(
        self,
        commission_type: str = "percent",
        commission_value: float = 0.0,
        slippage_mode: str = "ticks",
        slippage_value: float = 0.0,
        tick_size: float = 0.01,
        execution_timing: str = "next_open",
        tp_sl_priority: str = "stop_first",
        position_mode: str = "equity_pct",
        position_value: float = 1.0,
        pyramiding: int = 0,
        allow_same_bar_reentry: bool = False,
        initial_equity: float = 1.0,
    ):
        valid_execution = {"next_open", "close"}
        valid_priority = {"stop_first", "target_first"}
        valid_position = {"equity_pct", "fixed_qty"}
        valid_commission = {"percent", "bps", "fixed"}
        valid_slippage = {"ticks", "bps", "percent"}

        if execution_timing not in valid_execution:
            raise ValueError(f"Unsupported execution_timing: {execution_timing}")
        if tp_sl_priority not in valid_priority:
            raise ValueError(f"Unsupported tp_sl_priority: {tp_sl_priority}")
        if position_mode not in valid_position:
            raise ValueError(f"Unsupported position_mode: {position_mode}")
        if commission_type not in valid_commission:
            raise ValueError(f"Unsupported commission_type: {commission_type}")
        if slippage_mode not in valid_slippage:
            raise ValueError(f"Unsupported slippage_mode: {slippage_mode}")
        if pyramiding != 0:
            raise ValueError("Pyramiding not supported in calibration engine")

        self.commission_type = commission_type
        self.commission_value = commission_value
        self.slippage_mode = slippage_mode
        self.slippage_value = slippage_value
        self.tick_size = tick_size
        self.execution_timing = execution_timing
        self.tp_sl_priority = tp_sl_priority
        self.position_mode = position_mode
        self.position_value = position_value
        self.pyramiding = pyramiding
        self.allow_same_bar_reentry = allow_same_bar_reentry
        self.initial_equity = initial_equity

    def run(
        self,
        df: pd.DataFrame,
        entry_signals: pd.Series,
        exit_signals: pd.Series,
        tp_pct: Optional[float] = None,
        sl_pct: Optional[float] = None,
        max_hold_bars: Optional[int] = None,
    ) -> TVBacktestResult:
        if df.empty:
            return TVBacktestResult([], [], self.initial_equity, 0.0, 0.0, 0.0, 0.0)

        df_local = df.copy()
        df_local.columns = [c.lower() for c in df_local.columns]
        for col in ("open", "high", "low", "close"):
            if col not in df_local.columns:
                df_local[col] = df_local["close"]

        entries = entry_signals.reindex(df_local.index).fillna(False).astype(bool)
        exits = exit_signals.reindex(df_local.index).fillna(False).astype(bool)

        equity = float(self.initial_equity)
        equity_curve: List[float] = []
        trades: List[TVTrade] = []
        total_fees = 0.0
        total_slippage = 0.0

        position = 0
        entry_idx: Optional[int] = None
        entry_price: Optional[float] = None
        qty: float = 0.0
        hold_bars = 0
        entry_fee = 0.0
        entry_slippage_cost = 0.0

        pending_entry_idx: Optional[int] = None
        pending_exit_idx: Optional[int] = None

        for i in range(len(df_local)):
            row = df_local.iloc[i]
            open_px = float(row["open"])
            high_px = float(row["high"])
            low_px = float(row["low"])
            close_px = float(row["close"])
            exited_this_bar = False

            if position == 0 and pending_entry_idx == i:
                entry_price, slippage = self._apply_slippage(open_px, side="buy")
                qty = self._calc_qty(entry_price, equity)
                entry_slippage_cost = slippage * qty
                total_slippage += entry_slippage_cost
                entry_fee = self._calc_commission(entry_price, qty)
                equity -= entry_fee
                total_fees += entry_fee
                position = 1
                entry_idx = i
                hold_bars = 0
                pending_entry_idx = None

            if position == 1 and pending_exit_idx == i:
                exit_price, slippage = self._apply_slippage(open_px, side="sell")
                exit_slippage_cost = slippage * qty
                total_slippage += exit_slippage_cost
                exit_fee = self._calc_commission(exit_price, qty)
                equity, trade = self._close_trade(
                    equity,
                    entry_idx,
                    i,
                    entry_price,
                    exit_price,
                    qty,
                    entry_fee,
                    exit_fee,
                    entry_slippage_cost + exit_slippage_cost,
                    reason="signal",
                    df_index=df_local.index,
                )
                total_fees += exit_fee
                trades.append(trade)
                position = 0
                entry_idx = None
                entry_price = None
                qty = 0.0
                hold_bars = 0
                entry_fee = 0.0
                entry_slippage_cost = 0.0
                pending_exit_idx = None
                exited_this_bar = True

            if position == 1 and not exited_this_bar:
                hold_bars += 1

            if position == 1 and entry_price is not None and tp_pct and sl_pct and not exited_this_bar:
                target_price = entry_price * (1.0 + tp_pct)
                stop_price = entry_price * (1.0 - sl_pct)
                hit_target = high_px >= target_price
                hit_stop = low_px <= stop_price
                if hit_target or hit_stop:
                    exit_price = None
                    reason = None
                    if hit_target and hit_stop:
                        if self.tp_sl_priority == "target_first":
                            exit_price = target_price
                            reason = "tp"
                        else:
                            exit_price = stop_price
                            reason = "sl"
                    elif hit_target:
                        exit_price = target_price
                        reason = "tp"
                    elif hit_stop:
                        exit_price = stop_price
                        reason = "sl"

                    if exit_price is not None:
                        exit_price, slippage = self._apply_slippage(exit_price, side="sell")
                        exit_slippage_cost = slippage * qty
                        total_slippage += exit_slippage_cost
                        exit_fee = self._calc_commission(exit_price, qty)
                        equity, trade = self._close_trade(
                            equity,
                            entry_idx,
                            i,
                            entry_price,
                            exit_price,
                            qty,
                            entry_fee,
                            exit_fee,
                            entry_slippage_cost + exit_slippage_cost,
                            reason=reason,
                            df_index=df_local.index,
                        )
                        total_fees += exit_fee
                        trades.append(trade)
                        position = 0
                        entry_idx = None
                        entry_price = None
                        qty = 0.0
                        hold_bars = 0
                        entry_fee = 0.0
                        entry_slippage_cost = 0.0
                        exited_this_bar = True

            if position == 1 and max_hold_bars is not None and not exited_this_bar:
                if hold_bars >= max_hold_bars:
                    exit_price, slippage = self._apply_slippage(close_px, side="sell")
                    exit_slippage_cost = slippage * qty
                    total_slippage += exit_slippage_cost
                    exit_fee = self._calc_commission(exit_price, qty)
                    equity, trade = self._close_trade(
                        equity,
                        entry_idx,
                        i,
                        entry_price,
                        exit_price,
                        qty,
                        entry_fee,
                        exit_fee,
                        entry_slippage_cost + exit_slippage_cost,
                        reason="time",
                        df_index=df_local.index,
                    )
                    total_fees += exit_fee
                    trades.append(trade)
                    position = 0
                    entry_idx = None
                    entry_price = None
                    qty = 0.0
                    hold_bars = 0
                    entry_fee = 0.0
                    entry_slippage_cost = 0.0
                    exited_this_bar = True

            if position == 1 and exits.iloc[i] and not exited_this_bar:
                if self.execution_timing == "close":
                    exit_price, slippage = self._apply_slippage(close_px, side="sell")
                    exit_slippage_cost = slippage * qty
                    total_slippage += exit_slippage_cost
                    exit_fee = self._calc_commission(exit_price, qty)
                    equity, trade = self._close_trade(
                        equity,
                        entry_idx,
                        i,
                        entry_price,
                        exit_price,
                        qty,
                        entry_fee,
                        exit_fee,
                        entry_slippage_cost + exit_slippage_cost,
                        reason="signal",
                        df_index=df_local.index,
                    )
                    total_fees += exit_fee
                    trades.append(trade)
                    position = 0
                    entry_idx = None
                    entry_price = None
                    qty = 0.0
                    hold_bars = 0
                    entry_fee = 0.0
                    entry_slippage_cost = 0.0
                    exited_this_bar = True
                else:
                    if i + 1 < len(df_local):
                        pending_exit_idx = i + 1

            if position == 0 and entries.iloc[i] and (self.allow_same_bar_reentry or not exited_this_bar):
                if self.execution_timing == "close":
                    entry_price, slippage = self._apply_slippage(close_px, side="buy")
                    qty = self._calc_qty(entry_price, equity)
                    entry_slippage_cost = slippage * qty
                    total_slippage += entry_slippage_cost
                    entry_fee = self._calc_commission(entry_price, qty)
                    equity -= entry_fee
                    total_fees += entry_fee
                    position = 1
                    entry_idx = i
                    hold_bars = 0
                else:
                    if i + 1 < len(df_local):
                        pending_entry_idx = i + 1

            if position == 1 and entry_price is not None:
                equity_curve.append(equity + qty * (close_px - entry_price))
            else:
                equity_curve.append(equity)

        if position == 1 and entry_price is not None and entry_idx is not None:
            close_px = float(df_local["close"].iloc[-1])
            exit_price, slippage = self._apply_slippage(close_px, side="sell")
            exit_slippage_cost = slippage * qty
            total_slippage += exit_slippage_cost
            exit_fee = self._calc_commission(exit_price, qty)
            equity, trade = self._close_trade(
                equity,
                entry_idx,
                len(df_local) - 1,
                entry_price,
                exit_price,
                qty,
                entry_fee,
                exit_fee,
                entry_slippage_cost + exit_slippage_cost,
                reason="eod",
                df_index=df_local.index,
            )
            total_fees += exit_fee
            trades.append(trade)
            equity_curve[-1] = equity

        mdd_pct = self._calc_mdd(equity_curve)
        total_return_pct = ((equity / self.initial_equity) - 1.0) * 100.0

        return TVBacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            final_equity=equity,
            total_return_pct=total_return_pct,
            mdd_pct=mdd_pct,
            total_fees=total_fees,
            total_slippage=total_slippage,
        )

    def _calc_qty(self, price: float, equity: float) -> float:
        if price <= 0:
            return 0.0
        if self.position_mode == "fixed_qty":
            return float(self.position_value)
        allocation = float(self.position_value)
        allocation = max(0.0, min(1.0, allocation))
        return (equity * allocation) / price

    def _apply_slippage(self, price: float, side: str) -> tuple[float, float]:
        slip = 0.0
        if self.slippage_mode == "ticks":
            slip = float(self.slippage_value) * float(self.tick_size)
        elif self.slippage_mode == "bps":
            slip = price * (float(self.slippage_value) / 10000.0)
        elif self.slippage_mode == "percent":
            slip = price * (float(self.slippage_value) / 100.0)

        if side == "buy":
            return price + slip, slip
        return price - slip, slip

    def _calc_commission(self, price: float, qty: float) -> float:
        if self.commission_type == "percent":
            return price * qty * (self.commission_value / 100.0)
        if self.commission_type == "bps":
            return price * qty * (self.commission_value / 10000.0)
        if self.commission_type == "fixed":
            return float(self.commission_value)
        return 0.0

    def _close_trade(
        self,
        equity: float,
        entry_idx: Optional[int],
        exit_idx: int,
        entry_price: Optional[float],
        exit_price: float,
        qty: float,
        entry_fee: float,
        exit_fee: float,
        slippage_cost: float,
        reason: str,
        df_index: pd.Index,
    ) -> tuple[float, TVTrade]:
        if entry_price is None or entry_idx is None:
            return equity, TVTrade("", "", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "invalid")

        pnl_gross = (exit_price - entry_price) * qty
        total_fees = entry_fee + exit_fee
        pnl_net = pnl_gross - total_fees
        equity += pnl_gross - exit_fee

        trade = TVTrade(
            entry_time=str(df_index[entry_idx].date()),
            exit_time=str(df_index[exit_idx].date()),
            entry_bar_idx=int(entry_idx),
            exit_bar_idx=int(exit_idx),
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            qty=float(qty),
            pnl_gross=float(pnl_gross),
            pnl_net=float(pnl_net),
            fees=float(total_fees),
            slippage=float(slippage_cost),
            reason=reason,
        )
        return equity, trade

    @staticmethod
    def _calc_mdd(equity_curve: List[float]) -> float:
        if not equity_curve:
            return 0.0
        equity_arr = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr / peak) - 1.0
        return abs(float(drawdown.min()) * 100.0)
