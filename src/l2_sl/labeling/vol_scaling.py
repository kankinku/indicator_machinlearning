from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def get_vol_ewma(prices: pd.Series, span: int = 100) -> pd.Series:
    """
    Exponential Weighted Moving Average for Volatility.
    """
    # Simply use percentage returns
    returns = prices.pct_change()
    # EWM standard deviation
    vol = returns.ewm(span=span).std()
    return vol


def get_triple_barrier_events(
    close: pd.Series,
    t_events: pd.Index,
    pt_sl: List[float],
    target: pd.Series,
    min_ret: float = 0.0,
    vertical_barrier_times: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Getting the time of the first touch of the barriers.
    - pt_sl: [profit_take, stop_loss] multipliers of target (volatility).
    - target: volatility series (dynamic barrier width).
    """
    # 1) Get target barrier width (vol * k)
    # Align target with events
    target = target.reindex(t_events)
    target = target[target > min_ret]  # Filter out very low vol events if needed

    if vertical_barrier_times is None:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events)

    out_events = []
    
    # We loop through each event (start time)
    for loc in target.index:
        # Determine the end time for this path search (vertical barrier)
        t1 = vertical_barrier_times[loc]
        
        # Slice the price series for the window
        # If t1 is NaT, we go to end. If t1 exists, we go up to t1 (inclusive)
        if pd.isnull(t1):
            df0 = close[loc:]
        else:
            df0 = close[loc:t1]
            
        trgt = target[loc]
        
        start_price = close[loc]
        if pd.isna(start_price):
            continue
            
        # Thresholds (Dynamic)
        upper_barrier = start_price * (1 + pt_sl[0] * trgt)
        lower_barrier = start_price * (1 - pt_sl[1] * trgt)
        
        # Find touch times
        # df0 includes start time, so we should skip the first point to avoid instantaneous logic if needed
        # But usually we want forward looking, so loc is t=0.
        
        # Vectorized search within the window
        # Time of first upper touch
        u_touches = df0[df0 >= upper_barrier].index
        u_touch = u_touches[0] if len(u_touches) > 0 else pd.NaT
        
        # Time of first lower touch
        d_touches = df0[df0 <= lower_barrier].index
        d_touch = d_touches[0] if len(d_touches) > 0 else pd.NaT
        
        first_touch = pd.NaT
        event_type = 0 # 0: t1 (vertical), 1: upper, -1: lower
        
        # Logic: First touch wins
        if pd.isnull(u_touch) and pd.isnull(d_touch):
            # No barrier touch -> Vertical barrier logic
            first_touch = t1
            event_type = 0
        elif pd.isnull(u_touch):
            # Only down touch exists?
            # Check if it happened before t1
            if pd.notnull(t1) and d_touch > t1:
                first_touch = t1
                event_type = 0
            else:
                first_touch = d_touch
                event_type = -1
        elif pd.isnull(d_touch):
            # Only up touch exists
            if pd.notnull(t1) and u_touch > t1:
                first_touch = t1
                event_type = 0
            else:
                first_touch = u_touch
                event_type = 1
        else:
            # Both touched, see which is first
            if u_touch < d_touch:
                if pd.notnull(t1) and u_touch > t1:
                    first_touch = t1
                    event_type = 0
                else:
                    first_touch = u_touch
                    event_type = 1
            else:
                if pd.notnull(t1) and d_touch > t1:
                    first_touch = t1
                    event_type = 0
                else:
                    first_touch = d_touch
                    event_type = -1
                    
        out_events.append([loc, first_touch, trgt, event_type])

    out_df = pd.DataFrame(out_events, columns=["t0", "t1", "trgt", "event_type"])
    out_df.set_index("t0", inplace=True)
    return out_df


def generate_triple_barrier_labels(
    prices: pd.Series,
    k_up: float = 1.0,
    k_down: float = 1.0,
    horizon_bars: int = 20,
    min_ret: float = 1e-5,
    vol_span: int = 100
) -> pd.DataFrame:
    """
    Main entry point for Vol-Scaling Triple Barrier Labeling.
    
    Args:
        prices: Time-series of close prices. (Must be datetime indexed or compatible)
        k_up: Multiplier for upper barrier (volatility units).
        k_down: Multiplier for lower barrier.
        horizon_bars: Vertical barrier in number of bars.
        min_ret: Filter for volatility.
        vol_span: Span for EWMA volatility.
        
    Returns:
        DataFrame with index (t0), columns [label, t1, trgt]
        label: 1 (up), -1 (down), 0 (vertical barrier expired)
    """
    if not isinstance(prices.index, (pd.DatetimeIndex, pd.RangeIndex)):
        # If it's not a standard index, reset or warn. For now assume it works with slicing.
        pass

    # 1. Compute Volatility
    vol = get_vol_ewma(prices, span=vol_span)
    
    # 2. Define Vertical Barriers (t1)
    # t1 is simply the timestamp H bars ahead
    t1_series = prices.index.to_series().shift(-horizon_bars)
    
    # 3. Get Events (Triple Barrier)
    # We only compute for where we have enough data (drop last H bars basically) or handle NaNs
    valid_events = prices.index[:-horizon_bars]
    
    events = get_triple_barrier_events(
        close=prices,
        t_events=valid_events,
        pt_sl=[k_up, k_down],
        target=vol,
        min_ret=min_ret,
        vertical_barrier_times=t1_series
    )
    
    # Return formatted labels
    return events[["event_type", "t1", "trgt"]].rename(columns={"event_type": "label"})
