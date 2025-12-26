from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RegimeState:
    trend_score: float
    vol_level: float
    corr_score: float
    shock_flag: bool
    label: str = "NEUTRAL"
    hour: int = 0
    session_id: int = 0 # 0: Asia, 1: London, 2: NY, 3: Wrap


@dataclass
class PerformanceState:
    cpcv_mean: float
    cpcv_worst: float
    cpcv_std: float
    pbo: float
    dd: float
    turnover: float


@dataclass
class BudgetState:
    risk_budget: Dict[str, float]
    remaining_experiments: int


@dataclass
class MetaState:
    regime_state: RegimeState
    performance_state: PerformanceState
    budget_state: BudgetState
    recent_reason_codes: List[str] = field(default_factory=list)

