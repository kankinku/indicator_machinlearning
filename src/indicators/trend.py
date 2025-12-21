"""
추세성 지표 포팅: SMA/EMA/RMA/MSUM/MSTD/Typical Price/MACD.
원본: https://github.com/cinar/indicatorts (TypeScript)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .helpers import add, divide_by, subtract, multiply_by


@dataclass
class SMAConfig:
    period: int = 2


def sma(values: Sequence[float], config: SMAConfig | Dict[str, int] | None = None) -> List[float]:
    period = (config.period if isinstance(config, SMAConfig) else (config or {}).get("period", 2))
    result: List[float] = [0.0 for _ in values]
    window_sum = 0.0
    for i, v in enumerate(values):
        window_sum += v
        if i >= period:
            window_sum -= values[i - period]
            result[i] = window_sum / period
        else:
            result[i] = window_sum / (i + 1)
    return result


@dataclass
class EMAConfig:
    period: int = 12


def ema(values: Sequence[float], config: EMAConfig | Dict[str, int] | None = None) -> List[float]:
    period = (config.period if isinstance(config, EMAConfig) else (config or {}).get("period", 12))
    result: List[float] = [0.0 for _ in values]
    if not values:
        return result
    k = 2 / (1 + period)
    m = 1 - k
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = values[i] * k + result[i - 1] * m
    return result


@dataclass
class RMAConfig:
    period: int = 4


def rma(values: Sequence[float], config: RMAConfig | Dict[str, int] | None = None) -> List[float]:
    period = (config.period if isinstance(config, RMAConfig) else (config or {}).get("period", 4))
    result: List[float] = [0.0 for _ in values]
    window_sum = 0.0
    for i, v in enumerate(values):
        count = i + 1
        if i < period:
            window_sum += v
        else:
            window_sum = result[i - 1] * (period - 1) + v
            count = period
        result[i] = window_sum / count
    return result


@dataclass
class MSumConfig:
    period: int = 4


def msum(values: Sequence[float], config: MSumConfig | Dict[str, int] | None = None) -> List[float]:
    period = (config.period if isinstance(config, MSumConfig) else (config or {}).get("period", 4))
    result: List[float] = [0.0 for _ in values]
    window_sum = 0.0
    for i, v in enumerate(values):
        window_sum += v
        if i >= period:
            window_sum -= values[i - period]
        result[i] = window_sum
    return result


@dataclass
class MSTDConfig:
    period: int = 4


def mstd(values: Sequence[float], config: MSTDConfig | Dict[str, int] | None = None) -> List[float]:
    period = (config.period if isinstance(config, MSTDConfig) else (config or {}).get("period", 4))
    result: List[float] = [0.0 for _ in values]
    averages = sma(values, {"period": period})
    for i in range(len(values)):
        if i >= period - 1:
            acc = 0.0
            for k in range(i - (period - 1), i + 1):
                acc += (values[k] - averages[i]) ** 2
            result[i] = (acc / period) ** 0.5
    return result


def typical_price(highs: Sequence[float], lows: Sequence[float], closings: Sequence[float]) -> List[float]:
    return divide_by(3, add(add(highs, lows), closings))


@dataclass
class MACDConfig:
    fast: int = 12
    slow: int = 26
    signal: int = 9


def macd(closings: Sequence[float], config: MACDConfig | Dict[str, int] | None = None) -> Dict[str, List[float]]:
    if isinstance(config, MACDConfig):
        fast, slow, signal = config.fast, config.slow, config.signal
    else:
        cfg = config or {}
        fast, slow, signal = cfg.get("fast", 12), cfg.get("slow", 26), cfg.get("signal", 9)

    ema_fast = ema(closings, {"period": fast})
    ema_slow = ema(closings, {"period": slow})
    macd_line = subtract(ema_fast, ema_slow)
    signal_line = ema(macd_line, {"period": signal})
    return {"macd_line": macd_line, "signal_line": signal_line}

