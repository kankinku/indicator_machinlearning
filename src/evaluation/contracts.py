from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
from src.contracts import PolicySpec
from src.l1_judge.evaluator import SampleMetrics

@dataclass
class WindowResult:
    window_id: str
    raw_score: float
    median_score: float
    p10_score: float
    std_score: float
    violation_rate: float
    avg_alpha: float = 0.0
    normalized_score: float = 0.0
    complexity_score: float = 0.0

@dataclass
class BestSample:
    sample_id: str
    window_id: str
    risk_budget: Dict[str, float]
    metrics: SampleMetrics
    sample_score: float
    core: Dict[str, Any]
    # [V16] Artifact Components for One-Pass Persistence
    X_features: Optional[pd.DataFrame] = None
    window_df: Optional[pd.DataFrame] = None

# [V16] Alias/Upgrade for the new flow
@dataclass
class ModuleResult:
    policy_spec: PolicySpec
    module_key: str
    score: float
    window_results: List[WindowResult]
    best_sample: Optional[BestSample]
    stage: str
    # [V16] Single Source of Truth for Learning & Persistence
    reward_breakdown: Optional[Dict[str, Any]] = None
    fingerprint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # [vAlpha+] Economic Alpha Tags
    aos_score: float = 0.0
    is_economically_viable: bool = True
    viability_reason: str = ""

EvaluationResult = ModuleResult
