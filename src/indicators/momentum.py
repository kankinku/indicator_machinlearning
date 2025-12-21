"""
모멘텀 지표 포팅: RSI.
원본: https://github.com/cinar/indicatorts (TypeScript)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .trend import rma


@dataclass
class RSIConfig:
    period: int = 14


def rsi(closings: Sequence[float], config: RSIConfig | Dict[str, int] | None = None) -> List[float]:
    period = (config.period if isinstance(config, RSIConfig) else (config or {}).get("period", 14))
    gains: List[float] = [0.0 for _ in closings]
    losses: List[float] = [0.0 for _ in closings]

    for i in range(1, len(closings)):
        diff = closings[i] - closings[i - 1]
        if diff > 0:
            gains[i] = diff
            losses[i] = 0.0
        else:
            losses[i] = -diff
            gains[i] = 0.0

    mean_gains = rma(gains, {"period": period})
    mean_losses = rma(losses, {"period": period})

    r_value: List[float] = [0.0 for _ in closings]
    for i in range(1, len(closings)):
        rs = mean_gains[i] / mean_losses[i] if mean_losses[i] != 0 else float("inf")
        r_value[i] = 100 - 100 / (1 + rs)
    return r_value

