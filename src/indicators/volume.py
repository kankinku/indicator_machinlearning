"""
거래량 기반 지표 포팅: VWAP, OBV, MFI.
원본: https://github.com/cinar/indicatorts (TypeScript)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .helpers import (
    add_by,
    changes,
    check_same_length,
    divide,
    extract_signs,
    multiply,
    multiply_by,
    pow_all,
)
from .trend import msum, typical_price


@dataclass
class VWAPConfig:
    period: int = 14


def vwap(closings: Sequence[float], volumes: Sequence[float], config: VWAPConfig | Dict[str, int] | None = None) -> List[float]:
    check_same_length(closings, volumes)
    period = (config.period if isinstance(config, VWAPConfig) else (config or {}).get("period", 14))
    return divide(
        msum(multiply(closings, volumes), {"period": period}),
        msum(volumes, {"period": period}),
    )


def obv(closings: Sequence[float], volumes: Sequence[float]) -> List[float]:
    check_same_length(closings, volumes)
    result: List[float] = [0.0 for _ in closings]
    for i in range(1, len(closings)):
        result[i] = result[i - 1]
        if closings[i] > closings[i - 1]:
            result[i] += volumes[i]
        elif closings[i] < closings[i - 1]:
            result[i] -= volumes[i]
    return result


@dataclass
class MFIConfig:
    period: int = 14


def mfi(
    highs: Sequence[float],
    lows: Sequence[float],
    closings: Sequence[float],
    volumes: Sequence[float],
    config: MFIConfig | Dict[str, int] | None = None,
) -> List[float]:
    check_same_length(highs, lows, closings, volumes)
    period = (config.period if isinstance(config, MFIConfig) else (config or {}).get("period", 14))

    raw_money_flow = multiply(typical_price(highs, lows, closings), volumes)
    signs = extract_signs(changes(1, raw_money_flow))
    money_flow = multiply(signs, raw_money_flow)
    positive = [v if v >= 0 else 0.0 for v in money_flow]
    negative = [v if v < 0 else 0.0 for v in money_flow]

    money_ratio = divide(
        msum(positive, {"period": period}),
        msum(multiply_by(-1, negative), {"period": period}),
    )

    return add_by(100, multiply_by(-100, pow_all(add_by(1, money_ratio), -1)))

