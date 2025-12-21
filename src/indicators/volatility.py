"""
변동성 지표 포팅: True Range, ATR, Bollinger Bands.
원본: https://github.com/cinar/indicatorts (TypeScript)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .helpers import add, abs_values, check_same_length, max_rows, shift_right_and_fill_by, subtract
from .trend import sma, mstd


def true_range(highs: Sequence[float], lows: Sequence[float], closings: Sequence[float]) -> List[float]:
    check_same_length(highs, lows, closings)
    previous = shift_right_and_fill_by(1, closings[0], closings)
    return max_rows(
        subtract(highs, lows),
        abs_values(subtract(highs, previous)),
        abs_values(subtract(lows, previous)),
    )


@dataclass
class ATRConfig:
    period: int = 14


def atr(highs: Sequence[float], lows: Sequence[float], closings: Sequence[float], config: ATRConfig | Dict[str, int] | None = None) -> Dict[str, List[float]]:
    period = (config.period if isinstance(config, ATRConfig) else (config or {}).get("period", 14))
    tr_line = true_range(highs, lows, closings)
    atr_line = sma(tr_line, {"period": period})
    return {"tr_line": tr_line, "atr_line": atr_line}


@dataclass
class BBConfig:
    period: int = 20


def bollinger_bands(closings: Sequence[float], config: BBConfig | Dict[str, int] | None = None) -> Dict[str, List[float]]:
    period = (config.period if isinstance(config, BBConfig) else (config or {}).get("period", 20))
    std2 = [2 * s for s in mstd(closings, {"period": period})]
    middle = sma(closings, {"period": period})
    upper = add(middle, std2)
    lower = subtract(middle, std2)
    return {"upper": upper, "middle": middle, "lower": lower}
