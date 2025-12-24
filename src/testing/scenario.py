from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

@dataclass
class DataScenario:
    name: str
    behavior: str # "NO_TRADE", "HIGH_VOL", "PERFECT_UPTREND", "FLAT"
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        # Returns modified data or control signals
        if self.behavior == "PERFECT_UPTREND":
            # Manipulate prices to always go up
            new_close = np.linspace(100, 200, len(df))
            df = df.copy()
            df["close"] = new_close
        elif self.behavior == "NO_TRADE":
            # Set volume to zero or something that triggers filters
            if "volume" in df.columns:
                df["volume"] = 0.0
        return df

@dataclass
class PolicyScenario:
    name: str
    fixed_policies: List[Any] = field(default_factory=list)
    behavior: Optional[str] = None # "ALWAYS_FAIL_S1", "PASS_S1_FAIL_S2"

@dataclass
class EnvironmentScenario:
    name: str
    force_rigid: bool = False
    force_diversity_collapse: bool = False
    force_worker_exception: bool = False

@dataclass
class Scenario:
    name: str
    data: Optional[DataScenario] = None
    policy: Optional[PolicyScenario] = None
    env: Optional[EnvironmentScenario] = None
