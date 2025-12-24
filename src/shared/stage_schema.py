from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

@dataclass(frozen=True)
class StageSpec:
    """
    [V15] Curriculum Stage Specification (SSOT)
    Defines the hard gates, targets, and exploration parameters for a specific stage.
    """
    stage_id: int
    name: str
    
    # 1. Performance Gates (Hard/Soft)
    target_return_pct: float      # Annualized return target (%)
    alpha_floor: float            # Annualized excess return floor (vs Benchmark)
    min_trades_per_year: float    # Minimum activity level
    max_mdd_pct: float            # Survival limit
    min_profit_factor: float      # Quality gate
    
    # 2. Logic & Complexity Constraints
    and_terms_range: Tuple[int, int] = (1, 3)
    quantile_bias: str = "center" # "center", "tail", "spread"
    
    # 3. Walk-Forward Consistency
    wf_splits: int = 3
    wf_gate_mode: str = "soft"    # "soft" (penalty) or "hard" (rejection)
    
    # 4. Exploration Control
    exploration_slot: float = 0.2 # Ratio of samples dedicated to pure exploration in this stage
    
    # 5. Diagnostic Rules
    rejection_rate_range: Tuple[float, float] = (0.3, 0.9)
    min_pass_rate: float = 0.05

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict) -> 'StageSpec':
        # Remove keys not in dataclass to avoid errors
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
