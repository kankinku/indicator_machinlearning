
import pandas as pd
import numpy as np
from typing import List
from src.features.custom.base import CustomFeatureBase, CustomFeatureParam

class VariableMovingAverage(CustomFeatureBase):
    """
    Sample Custom Indicator: Variable Moving Average (VMA)
    This demonstrates how to add a new indicator without touching core files.
    """
    
    @property
    def id(self) -> str:
        return "VMA"
        
    @property
    def name(self) -> str:
        return "Variable Moving Average"
        
    @property
    def category(self) -> str:
        return "trend"
        
    @property
    def description(self) -> str:
        return "Moving average that automatically adjusts its smoothing sensitivity based on volatility."
        
    @property
    def params(self) -> List[CustomFeatureParam]:
        return [
            CustomFeatureParam(name="window", param_type="int", default=10, min=5, max=50),
            CustomFeatureParam(name="vi_window", param_type="int", default=10, min=5, max=30)
        ]
        
    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Parameters
        window = int(kwargs.get("window", 10))
        vi_window = int(kwargs.get("vi_window", 10))
        
        close = df["close"]
        
        # Calculate Volatility Index (VI)
        # Here we use a simple efficiency ratio (ER) as VI:
        # ER = Change / Sum of absolute changes
        change = close.diff(window).abs()
        volatility = close.diff().abs().rolling(window=vi_window).sum()
        
        # Avoid division by zero
        er = change / (volatility + 1e-9)
        er = er.fillna(0)
        
        # VMA Calculation
        # VMA = PrevVMA + constant * ER * (Close - PrevVMA)
        # We can implement this vectorised or using ewm with arguments, 
        # but efficient alpha EWM is supported in pandas via adjustments.
        # However, for true variable alpha, pandas ewm is static alpha.
        # We will use a python loop for this specific custom indicator (or numbda if available, but stick to pure python/pandas for compat)
        # Since this is a test, a loop is fine for reasonable data sizes.
        
        vma = np.zeros_like(close)
        vma[:] = np.nan
        
        # Initialize
        values = close.values
        er_vals = er.values
        
        # First valid index
        start_idx = max(window, vi_window)
        if start_idx >= len(values):
            return pd.DataFrame({"VMA": vma}, index=df.index)
            
        vma[start_idx-1] = values[start_idx-1] # Seed
        
        # SC = Smooth Constant. Let's assume standard 0.1 for base scaling
        sc = 2.0 / (window + 1)
        
        for i in range(start_idx, len(values)):
            # Alpha varies by ER
            alpha = er_vals[i] * sc
            vma[i] = vma[i-1] + alpha * (values[i] - vma[i-1])
            
        return pd.DataFrame({f"VMA_{window}_{vi_window}": vma}, index=df.index)
