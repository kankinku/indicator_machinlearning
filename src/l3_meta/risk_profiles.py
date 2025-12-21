from dataclasses import dataclass
from typing import List, Tuple

from src.config import config


@dataclass(frozen=True)
class RiskProfile:
    profile_id: str
    k_up_range: Tuple[float, float]
    k_down_range: Tuple[float, float]
    horizon_range: Tuple[int, int]


def _split_range(min_v: float, max_v: float, low_frac: float, high_frac: float) -> Tuple[float, float]:
    span = max_v - min_v
    return min_v + (span * low_frac), min_v + (span * high_frac)


def _split_int_range(min_v: int, max_v: int, low_frac: float, high_frac: float) -> Tuple[int, int]:
    low, high = _split_range(float(min_v), float(max_v), low_frac, high_frac)
    low_i = int(round(low))
    high_i = int(round(high))
    if low_i > high_i:
        low_i = high_i
    return low_i, high_i


def _clamp_range(min_v: float, max_v: float, low: float, high: float) -> Tuple[float, float]:
    low_c = max(min_v, low)
    high_c = min(max_v, high)
    if low_c > high_c:
        low_c = high_c
    return low_c, high_c


def _clamp_int_range(min_v: int, max_v: int, low: int, high: int) -> Tuple[int, int]:
    low_c = max(min_v, low)
    high_c = min(max_v, high)
    if low_c > high_c:
        low_c = high_c
    return low_c, high_c


def get_default_risk_profiles() -> List[RiskProfile]:
    k_up_min, k_up_max = config.RISK_K_UP_MIN, config.RISK_K_UP_MAX
    k_down_min, k_down_max = config.RISK_K_DOWN_MIN, config.RISK_K_DOWN_MAX
    h_min, h_max = config.RISK_HORIZON_MIN, config.RISK_HORIZON_MAX

    profile_defs = [
        ("SHORT_TERM", (0.0, 0.35), (0.0, 0.35), (0.0, 0.3)),
        ("BALANCED", (0.25, 0.7), (0.25, 0.7), (0.2, 0.6)),
        ("LONG_TERM", (0.6, 1.0), (0.6, 1.0), (0.5, 1.0)),
    ]

    profiles: List[RiskProfile] = []
    for profile_id, k_up_frac, k_down_frac, h_frac in profile_defs:
        k_up_range = _split_range(k_up_min, k_up_max, *k_up_frac)
        k_down_range = _split_range(k_down_min, k_down_max, *k_down_frac)
        h_range = _split_int_range(h_min, h_max, *h_frac)

        k_up_range = _clamp_range(k_up_min, k_up_max, *k_up_range)
        k_down_range = _clamp_range(k_down_min, k_down_max, *k_down_range)
        h_range = _clamp_int_range(h_min, h_max, *h_range)

        profiles.append(
            RiskProfile(
                profile_id=profile_id,
                k_up_range=k_up_range,
                k_down_range=k_down_range,
                horizon_range=h_range,
            )
        )

    return profiles
