from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_benchmark_return(
    prices: pd.Series,
    exposure_ratio: float,
    mode: str,
    fixed_return_pct: float,
) -> float:
    if prices is None or prices.empty:
        return 0.0

    start = float(prices.iloc[0])
    end = float(prices.iloc[-1])
    if start == 0.0:
        return 0.0

    bh_return = (end / start - 1.0) * 100.0
    exposure = float(np.clip(exposure_ratio, 0.0, 1.0))
    mode_key = (mode or "bh").lower().strip()

    if mode_key == "fixed":
        return float(fixed_return_pct)
    if mode_key == "cash":
        return 0.0
    if mode_key == "exposure":
        return bh_return * exposure
    if mode_key == "exposure_fixed":
        return float(fixed_return_pct) * exposure
    return bh_return


def compute_excess_return(
    total_return_pct: float,
    benchmark_return_pct: float,
) -> float:
    return float(total_return_pct) - float(benchmark_return_pct)
