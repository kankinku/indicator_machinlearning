from __future__ import annotations

import pandas as pd
import numpy as np
import ta

from src.l3_meta.state import RegimeState

class RegimeDetector:
    """
    Detects the current market regime based on technical indicators.
    Used by L3 Meta-Agent to select appropriate Strategy Templates.
    """

    def detect(self, df: pd.DataFrame) -> RegimeState:
        """
        Analyze the recent history of the DataFrame to determine regime.
        Incorporates Macro Data (VIX, US10Y) if available.
        """
        if len(df) < 50:
            return RegimeState(0.0, 1.0, 0.0, False, "SIDEWAYS")
            
        current_idx = -1
        
        # --- 1. Technical Factors ---
        # ADX & Trend
        try:
            adx = ta.trend.adx(df["high"], df["low"], df["close"], window=14).iloc[current_idx]
        except:
            adx = 0
        
        sma_50 = df["close"].rolling(50).mean().iloc[current_idx]
        sma_200 = df["close"].rolling(200).mean().iloc[current_idx]
        current_price = df["close"].iloc[current_idx]
        
        trend_direction = 0.0
        if current_price > sma_50: trend_direction += 0.5
        if sma_50 > sma_200: trend_direction += 0.5
        if current_price < sma_50: trend_direction -= 0.5
        if sma_50 < sma_200: trend_direction -= 0.5
            
        adx_factor = min(adx / 50.0, 1.0)
        final_trend_score = trend_direction * adx_factor
        
        # Volatility Ratio
        returns = df["close"].pct_change()
        vol_short = returns.rolling(20).std().iloc[current_idx]
        vol_long = returns.rolling(100).std().iloc[current_idx]
        vol_level = (vol_short / vol_long) if vol_long > 0 else 1.0
        
        # Shock Check (Technical)
        bb_l = ta.volatility.bollinger_lband(df["close"], window=20, window_dev=2.5).iloc[current_idx]
        shock_flag = current_price < bb_l
        
        # Correlation
        corr_score = 0.0
        if "volume" in df.columns:
            corr_score = df["close"].rolling(20).corr(df["volume"]).iloc[current_idx]
            if np.isnan(corr_score): corr_score = 0.0

        # --- 2. Macro Factors (New) ---
        # VIX Check
        vix = 20.0 # Default neutral
        if "vix_close" in df.columns:
            vix = df["vix_close"].iloc[current_idx]
            if vix == 0: vix = 20.0 # Fallback if data missing (0)

        # Yield Check (10Y)
        yield_10y = 4.0
        if "us10y_close" in df.columns:
            yield_10y = df["us10y_close"].iloc[current_idx]
            
        # --- 3. Advanced Classification ---
        # Logic: 
        # PANIC: VIX > 30 or Shock or Extreme Bear Trend
        # GOLDILOCKS: Trend > 0, VIX < 20, Yield stable (proxy)
        # STAGFLATION: Trend < 0, Yield > 4.5 (High), VIX Moderate? (Simplification)
        # BULL_RUN: Strong Trend (>0.5), VIX < 25
        
        # --- 3. Advanced Context Analysis ---
        # Credit Risk Proxy: Spread between LQD/HYG or SPY/HYG?
        # Ideally HYG (Junk) vs LQD (Grade). Lacking LQD, we check if HYG is crashing relative to SPY?
        # Let's use simple HYG trend as additional confirmation.
        hyg_trend = 0.0
        if "hyg_close" in df.columns:
            hyg_sma50 = df["hyg_close"].rolling(50).mean().iloc[current_idx]
            hyg_curr = df["hyg_close"].iloc[current_idx]
            hyg_trend = (hyg_curr - hyg_sma50) / hyg_sma50
            
        # Tech vs Energy (Growth vs Inflation Rotation)
        sector_ratio_trend = 0.0
        if "xlk_close" in df.columns and "xle_close" in df.columns:
            xlk = df["xlk_close"]
            xle = df["xle_close"]
            ratio = xlk / (xle + 1e-9)
            ratio_sma = ratio.rolling(50).mean().iloc[current_idx]
            curr_ratio = ratio.iloc[current_idx]
            sector_ratio_trend = (curr_ratio - ratio_sma) / ratio_sma # > 0 means Tech leading
            
        # --- 4. Regime Classification ---
        label = "SIDEWAYS"
        
        # Conditions
        is_panic = (vix > 30) or shock_flag or (final_trend_score < -0.7) or (hyg_trend < -0.05) # Credit Crash
        is_bull_run = (final_trend_score > 0.4) and (vix < 25)
        is_goldilocks = (final_trend_score > 0) and (vix < 20) and (yield_10y < 4.0)
        is_stagflation = (final_trend_score < -0.1) and ((yield_10y > 4.2) or (sector_ratio_trend < -0.05)) # High Rates OR Energy Leading
        
        if is_panic:
            label = "PANIC"
        elif is_stagflation:
            label = "STAGFLATION"
        elif is_bull_run:
            # Check for Sector Confirmation? 
            # If Tech is leading, it's a Growth Bull. 
            label = "BULL_RUN"
        elif is_goldilocks:
            label = "GOLDILOCKS"
        elif vol_level > 1.5:
            label = "HIGH_VOL"
        elif final_trend_score < -0.2:
            label = "BEAR_TREND"
        else:
            label = "SIDEWAYS"
            
        return RegimeState(
            trend_score=float(final_trend_score),
            vol_level=float(vol_level),
            corr_score=float(corr_score),
            shock_flag=bool(shock_flag),
            label=label
        )
