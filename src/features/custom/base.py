
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class CustomFeatureParam:
    name: str
    param_type: str # int, float, choice
    default: Any
    min: Any = None
    max: Any = None
    step: Any = None
    choices: list = None

class CustomFeatureBase(ABC):
    """Base class for all custom dynamically loaded indicators."""
    
    @property
    @abstractmethod
    def id(self) -> str:
        pass
        
    @property
    @abstractmethod
    def name(self) -> str:
        pass
        
    @property
    @abstractmethod
    def category(self) -> str:
        pass # momentum, volatility, trend, volume
        
    @property
    @abstractmethod
    def description(self) -> str:
        pass
        
    @property
    @abstractmethod
    def params(self) -> List[CustomFeatureParam]:
        pass
        
    @abstractmethod
    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute the indicator.
        Args:
            df: OHLCV DataFrame
            **kwargs: Parameters defined in self.params
        Returns:
            DataFrame with feature columns
        """
        pass
