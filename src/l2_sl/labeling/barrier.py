
import numpy as np
import pandas as pd
from typing import Tuple, Optional

def get_triple_barrier_labels(
    close_prices: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: Tuple[float, float],
    target: pd.Series,
    min_ret: float = 0.005,
    max_window: int = 20
) -> pd.Series:
    """
    Implements the Triple Barrier Method (De Prado).
    
    Args:
        close_prices (pd.Series): Time series of close prices.
        t_events (pd.DatetimeIndex): Timestamps of events (entries) to label.
        pt_sl (Tuple[float, float]): (Profit Taking multiplier, Stop Loss multiplier).
                                     e.g., (1, 1) means symmetric barriers relative to volatility.
        target (pd.Series): Volatility target (daily standard deviation). Used to set barrier width.
        min_ret (float): Minimum target return required to run the barrier search.
        max_window (int): Vertical barrier (max holding period in days).

    Returns:
        pd.Series: Labels (-1, 0, 1) for each event in t_events.
                   1: Hits Upper Barrier (Profit) first.
                   -1: Hits Lower Barrier (Loss) first.
                   0: Time out (Vertical Barrier) - Can be mapped to 0 or sign of return.
    """
    # 1. Setup Barriers
    out = pd.DataFrame(index=t_events)
    
    # Upper/Lower targets based on vol * multiplier
    # To simplify, we look ahead 'max_window' for each point
    
    events = t_events
    labels = []
    
    for t0 in events:
        try:
            # Slice the future window
            t1 = t0 + pd.Timedelta(days=max_window*2) # Buffer for non-trading days
            # Get path
            path = close_prices.loc[t0:t1]
            if path.empty:
                labels.append(0)
                continue
                
            # Limit to max_window trading bars
            if len(path) > max_window:
                path = path.iloc[:max_window+1]
                
            # Initial Price
            p0 = path.iloc[0]
            
            # Volatility at t0
            vol = target.loc[t0] if t0 in target.index else min_ret
            # Ensure min volatility
            vol = max(vol, min_ret)
            
            # Thresholds
            upper = p0 * (1 + pt_sl[0] * vol)
            lower = p0 * (1 - pt_sl[1] * vol)
            
            # Find touch times
            # Touched Upper?
            touch_upper = path[path >= upper].index.min()
            # Touched Lower?
            touch_lower = path[path <= lower].index.min()
            
            label = 0 # Default: Time out
            
            if pd.isna(touch_upper) and pd.isna(touch_lower):
                # Vertical Barrier Hit (Time Out)
                # Option: Return binary based on sign? Or 0?
                # De Prado suggests keeping 0 or label based on return sign if wanted.
                # Let's use sign of return at end
                label = 0
                
            elif pd.isna(touch_lower):
                # Only Upper hit
                label = 1
                
            elif pd.isna(touch_upper):
                # Only Lower hit
                label = -1
                
            else:
                # Both hit, check which first
                if touch_upper < touch_lower:
                    label = 1
                else:
                    label = -1
            
            labels.append(label)
            
        except KeyError:
            labels.append(0)
            
    out['label'] = labels
    return out['label']

def compute_daily_volatility(close: pd.Series, span: int = 50) -> pd.Series:
    """
    Computes daily volatility using EWMA of returns.
    """
    # Simply: daily return std dev
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    
    # Simple Percentage Return
    ret = close.pct_change()
    # EWM Std Dev
    std = ret.ewm(span=span).std()
    return std
