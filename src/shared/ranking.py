from __future__ import annotations

import math
from typing import Optional, Tuple

MIN_TRADES = 20
MIN_WIN_RATE = 0.5
MIN_CPCV_MEAN = 0.0
MIN_CPCV_WORST = 0.0
MAX_VOL_RATIO = 3.0


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        val = float(value)
        if math.isnan(val):
            return default
        return val
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def compute_vol_ratio(cpcv_mean: float, cpcv_std: float) -> float:
    denom = abs(cpcv_mean) + 1e-9
    return cpcv_std / denom


def check_return_stability(
    cpcv_mean: float,
    cpcv_std: float,
    cpcv_worst: float,
    win_rate: float,
    n_trades: int,
    min_trades: int = MIN_TRADES,
    min_win_rate: float = MIN_WIN_RATE,
    min_cpcv_mean: float = MIN_CPCV_MEAN,
    min_cpcv_worst: float = MIN_CPCV_WORST,
    max_vol_ratio: float = MAX_VOL_RATIO,
) -> Tuple[bool, float]:
    mean_v = _coerce_float(cpcv_mean)
    std_v = _coerce_float(cpcv_std)
    worst_v = _coerce_float(cpcv_worst)
    win_v = _coerce_float(win_rate)
    trades_v = _coerce_int(n_trades)

    vol_ratio = compute_vol_ratio(mean_v, std_v)
    stability_pass = (
        trades_v >= min_trades
        and win_v >= min_win_rate
        and mean_v > min_cpcv_mean
        and worst_v >= min_cpcv_worst
        and vol_ratio <= max_vol_ratio
    )
    return stability_pass, vol_ratio


def return_rank_key(
    total_return: Optional[float],
    stability_pass: bool,
    win_rate: float,
    n_trades: int,
    vol_ratio: float,
) -> Tuple[float, float, float, float, float]:
    safe_return = _coerce_float(total_return, default=float("-inf"))
    return (
        1.0 if stability_pass else 0.0,
        safe_return,
        _coerce_float(win_rate),
        float(_coerce_int(n_trades)),
        -_coerce_float(vol_ratio, default=float("inf")),
    )
