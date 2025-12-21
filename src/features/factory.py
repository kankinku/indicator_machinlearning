
from __future__ import annotations

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import ta
from .custom.loader import loader
from src.config import config
from src.features.registry import FeatureRegistry

# Initialize loader once
try:
    loader.load_all()
except Exception as e:
    print(f"Warning: Failed to load custom indicators: {e}")

class FeatureFactory:
    """
    V2 Feature Factory: Generates features from a Genome (Dynamic Recipe).
    """
    def __init__(self):
        self.registry = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH))
        self.registry.initialize()
        # Cache for compiled feature classes to avoid re-executing code snippets every time
        self._execution_cache: Dict[str, Any] = {}

    def generate_from_genome(self, df: pd.DataFrame, genome: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate features based on the provided genome using the Dynamic Registry.
        """
        if df.empty:
            return pd.DataFrame()
            
        # Standardize columns
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        feature_chunks = []
        
        for feature_id, params in genome.items():
            try:
                # Dynamic Generation Only
                chunk = self._try_dynamic_generation(feature_id, df, params)
                
                if chunk is not None and not chunk.empty:
                    feature_chunks.append(chunk)
                else:
                    # Log warning but continue
                    # print(f"Warning: Feature {feature_id} failed to generate (Returned None/Empty).")
                    pass
                    
            except Exception as e:
                print(f"Error generating {feature_id}: {e}")
                continue

        if not feature_chunks:
            return pd.DataFrame(index=df.index)
            
        features = pd.concat(feature_chunks, axis=1)
        
        # Handle Inf/Nan
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().fillna(0.0)
        
        return features

    def _try_dynamic_generation(self, feature_id: str, df: pd.DataFrame, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Attempts to generate feature using the Dynamic Registry.
        """
        # 1. Check Registry
        metadata = self.registry.get(feature_id)
        if not metadata:
            # If not found, we no longer have a fallback.
            # This enforces that ALL features must be in the registry.
            # print(f"Registry Miss: {feature_id}") 
            return None
            
        # 2. Check Execution Cache
        handler_cls = self._execution_cache.get(feature_id)
        
        if not handler_cls:
            # 3. Dynamic Compilation
            local_scope = {}
            try:
                exec(metadata.code_snippet, globals(), local_scope)
                
                if metadata.handler_func not in local_scope:
                    raise ValueError(f"Handler {metadata.handler_func} not found in snippet for {feature_id}")
                    
                handler_cls = local_scope[metadata.handler_func]
                self._execution_cache[feature_id] = handler_cls
            except Exception as e:
                print(f"Compilation Error for {feature_id}: {e}")
                return None
        
        # 4. Instantiate and Compute
        try:
            instance = handler_cls()
            result = instance.compute(df, **params)
            return result
        except Exception as e:
            print(f"Execution Error for {feature_id}: {e}")
            raise e
