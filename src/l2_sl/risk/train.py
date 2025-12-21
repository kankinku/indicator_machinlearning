from __future__ import annotations

from typing import Dict, Iterable, List, Any
import numpy as np
import pandas as pd

def train_risk_model(vol_estimates: Iterable[float]) -> Dict[str, Any]:
    """
    Train a Risk Model (e.g. VaR / Tail Risk estimator).
    For this vibe impl, we simply estimate the 95th and 99th percentile of the provided volatility/risk metric.
    
    Args:
        vol_estimates: Historical volatility or risk scores.
        
    Returns:
        Parameter dict (thresholds).
    """
    values = list(vol_estimates)
    if not values:
        return {"avg_vol": 0.0, "p95": 0.0, "p99": 0.0}
        
    arr = np.array(values)
    
    return {
        "avg_vol": float(np.mean(arr)),
        "std_vol": float(np.std(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max_vol": float(np.max(arr))
    }
