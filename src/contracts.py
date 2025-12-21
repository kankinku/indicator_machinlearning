from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ValidationError(Exception):
    """Raised when a contract is violated."""


@dataclass
class TunableParamSpec:
    name: str
    param_type: str  # "float", "int", "categorical"
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None # Added for evolution granularity
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None

    def validate(self, value: Any) -> Any:
        if self.param_type == "categorical":
            if self.choices is None:
                raise ValidationError(f"{self.name} is categorical but no choices provided")
            if value not in self.choices:
                raise ValidationError(f"{self.name} must be one of {self.choices}, got {value}")
            return value

        if value is None:
            raise ValidationError(f"{self.name} requires a value")

        if self.param_type not in ("float", "int"):
            raise ValidationError(f"{self.name} has unsupported type {self.param_type}")

        if self.min is not None and value < self.min:
            raise ValidationError(f"{self.name} below min {self.min}: {value}")
        if self.max is not None and value > self.max:
            raise ValidationError(f"{self.name} above max {self.max}: {value}")

        if self.param_type == "int" and not isinstance(value, int):
            raise ValidationError(f"{self.name} must be int, got {type(value).__name__}")
        if self.param_type == "float" and not isinstance(value, (int, float)):
            raise ValidationError(f"{self.name} must be float, got {type(value).__name__}")
        return value


@dataclass
class StrategyTemplate:
    template_id: str
    name: str
    required_features: List[str]
    labeling_family: str
    model_family: str
    risk_family: str
    tunable_params_schema: List[TunableParamSpec]
    base_constraints: Dict[str, Any] = field(default_factory=dict)

    def validate_params(self, tuned_params: Dict[str, Any]) -> Dict[str, Any]:
        return tuned_params


@dataclass
class PolicySpec:
    spec_id: str
    
    # [Changed] Instead of just template_id, we now have a full Genome.
    # If the Agent uses a predefined template, we can fill this genome from the template.
    # Format: { "RSI": {"window": 14}, "MACD": {"fast": 12...} }
    feature_genome: Dict[str, Dict[str, Any]]
    
    # Meta-params still exist for execution & risk
    risk_budget: Dict[str, Any]
    execution_assumption: Dict[str, Any]
    
    # Legacy support (Optional)
    template_id: str = "GENOME"
    tuned_params: Dict[str, Any] = field(default_factory=dict) # Legacy param slot
    data_window: Dict[str, Any] = field(default_factory=dict)
    rl_meta: Dict[str, Any] = field(default_factory=dict)

    def validate_genome(self) -> None:
        # TODO: Validate against Universe
        pass

    def validate_against_template(self, template: StrategyTemplate) -> None:
        pass


@dataclass
class Forecast:
    p_dir: Dict[str, float]
    p_dir_calibrated: Dict[str, float]
    risk_pred: Dict[str, float]
    uncertainty: Dict[str, float]


@dataclass
class FixSuggestion:
    suggested_param_changes: Dict[str, Any] = field(default_factory=dict)
    suggested_constraints: Dict[str, Any] = field(default_factory=dict)
    suggested_template_switch: Optional[str] = None


@dataclass
class Verdict:
    approved: bool
    hard_fail_codes: List[str] = field(default_factory=list)
    soft_warn_codes: List[str] = field(default_factory=list)
    scorecard: Dict[str, Any] = field(default_factory=dict)
    fix_suggestion: Optional[FixSuggestion] = None
    
    # [New] Inspectable Logic (Why did we trade?)
    trade_logic: Optional[Dict[str, Any]] = None 


@dataclass
class LedgerRecord:
    exp_id: str
    timestamp: float
    
    policy_spec: PolicySpec
    
    data_hash: str
    feature_hash: str
    label_hash: str
    
    model_artifact_ref: str
    
    cpcv_metrics: Dict[str, Any]
    pbo: float
    risk_report: Dict[str, Any]
    
    reason_codes: List[str]
    fix_suggestion: Optional[FixSuggestion] = None
    
    # [New] Store the Verdict directly or parts of it for easy query
    verdict_dump: Optional[Dict[str, Any]] = None

# Alias for compatibility with scripts
FeatureParam = TunableParamSpec

@dataclass
class FeatureMetadata:
    """
    Metadata for a dynamically registered feature (indicator).
    Acts as the DNA structure in the evolution system.
    """
    feature_id: str
    name: str
    category: str  # e.g., "momentum", "trend", "volatility"
    
    # Implementation details
    description: str
    code_snippet: str  # Python code defining the logic (usually a class or function)
    handler_func: str  # Name of the function/class to call in the snippet
    
    # Configuration
    params: List[TunableParamSpec]
    complexity_score: float = 1.0  # 1.0 = Standard, Higher = More complex/expensive
    
    # Evolution Metadata
    source: str = "builtin"  # e.g., "builtin", "github_crawler", "mutation"
    tags: List[str] = field(default_factory=list)  # e.g., ["regime_bull", "short_term"]
    
    # Stats (updated by L1 Judge)
    fitness_score: float = 0.0
    usage_count: int = 0

