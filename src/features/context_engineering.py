
import pandas as pd
import numpy as np
from typing import List

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes temporal information into cyclical features (Sin/Cos),
    allowing the model to learn seasonality (Monthly, Weekly, Yearly patterns).
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to convert if not index
        if 'date' in df.columns:
            dt_series = pd.to_datetime(df['date'])
        else:
            return df # Cannot generate without datetime
    else:
        dt_series = df.index.to_series()

    # 1. Day of Week (0-6) -> Weekly Cycle
    dow = dt_series.dt.dayofweek
    df['time_dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['time_dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # 2. Month (1-12) -> Monthly/Yearly Seasonality
    month = dt_series.dt.month
    df['time_month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    df['time_month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)

    # 3. Day of Year (1-365) -> Annual Cycle
    doy = dt_series.dt.dayofyear
    df['time_doy_sin'] = np.sin(2 * np.pi * (doy - 1) / 365)
    df['time_doy_cos'] = np.cos(2 * np.pi * (doy - 1) / 365)
    
    # 4. Week of Year (Proxy for Quarter/Earnings Season)
    # week = dt_series.dt.isocalendar().week
    # df['time_woy_sin'] = np.sin(2 * np.pi * (week - 1) / 52)

    return df

def add_relative_features(df: pd.DataFrame, target_col: str, context_cols: List[str]) -> pd.DataFrame:
    """
    Generates relative strength ratios between the target asset and context assets.
    Example: QQQ / SPY, QQQ / VIX, QQQ / TLT
    """
    df = df.copy()
    if target_col not in df.columns:
        return df
        
    for ctx in context_cols:
        if ctx in df.columns:
            # Avoid division by zero
            denom = df[ctx].replace(0, np.nan).ffill()
            
            # Ratio
            df[f'rel_{ctx}_ratio'] = df[target_col] / (denom + 1e-9)
            
            # Correlation (Rolling 20) - Are they moving together or diverging?
            # df[f'rel_{ctx}_corr20'] = df[target_col].rolling(20).corr(df[ctx]).fillna(0)
            
    return df

def add_statistical_features(df: pd.DataFrame, target_col: str, windows=[20, 60]) -> pd.DataFrame:
    """
    Adds higher-order statistical moments (Skew, Kurtosis) to detect tail risks.
    """
    df = df.copy()
    for w in windows:
        # Skewness: Asymmetry of returns (Negative skew = crash risk)
        df[f'stat_skew_{w}'] = df[target_col].rolling(window=w).skew().fillna(0)
        
        # Kurtosis: Fat tails (High kurtosis = extreme event risk)
        df[f'stat_kurt_{w}'] = df[target_col].rolling(window=w).kurt().fillna(0)
        
        # Z-Score: How extreme is the current price relative to recent history?
        mean = df[target_col].rolling(window=w).mean()
        std = df[target_col].rolling(window=w).std()
        df[f'stat_zscore_{w}'] = (df[target_col] - mean) / (std + 1e-9)
        
    return df.fillna(0)
