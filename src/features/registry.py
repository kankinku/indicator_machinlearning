import os
import json
from typing import Dict, List, Optional
from threading import Lock

from src.contracts import FeatureMetadata, TunableParamSpec, ValidationError
from src.shared.logger import get_logger

logger = get_logger("feature.registry")

class FeatureRegistry:
    """
    Central One Source of Truth for all available features (indicators).
    Manages the lifecycle of features: Registration, Retrieval, Persistence.
    
    Adheres to:
    - One Source of Truth (Single JSON file DB)
    - Thread Safety (Locking)
    """
    
    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self._features: Dict[str, FeatureMetadata] = {}
        self._lock = Lock()
        self._loaded = False
        
    def initialize(self):
        """Loads the registry from disk."""
        with self._lock:
            if self._loaded:
                return
            
            self._load_from_disk()
            self._loaded = True
            
    def register(self, metadata: FeatureMetadata, overwrite: bool = False):
        """
        Registers a new feature into the ecosystem.
        """
        with self._lock:
            if not overwrite and metadata.feature_id in self._features:
                raise ValueError(f"Feature {metadata.feature_id} already exists.")
            
            self._features[metadata.feature_id] = metadata
            self._save_to_disk()
            logger.info(f"Registered feature: {metadata.feature_id} ({metadata.name})")

    def get(self, feature_id: str) -> Optional[FeatureMetadata]:
        if not self._loaded:
            self.initialize()
        return self._features.get(feature_id)

    def list_all(self) -> List[FeatureMetadata]:
        if not self._loaded:
            self.initialize()
        return list(self._features.values())
        
    def list_by_category(self, category: str) -> List[FeatureMetadata]:
        if not self._loaded:
            self.initialize()
        return [f for f in self._features.values() if f.category == category]

    def _load_from_disk(self):
        if not os.path.exists(self.registry_path):
            self._features = {}
            logger.warning(f"Registry file not found at {self.registry_path}. Starting empty.")
            return

        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                # Reconstruct generic dictionary to Dataclass
                # Need to deserialize nested TunableParamSpec
                params = []
                for p in item.get('params', []):
                    params.append(TunableParamSpec(**p))
                
                item['params'] = params
                meta = FeatureMetadata(**item)
                self._features[meta.feature_id] = meta
                
            logger.info(f"Loaded {len(self._features)} features from registry.")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            raise e

    def _save_to_disk(self):
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        
        serializable_data = []
        for feature in self._features.values():
            # Dataclass to dict
            f_dict = feature.__dict__.copy()
            # Handle nested objects (TunableParamSpec)
            f_dict['params'] = [p.__dict__ for p in feature.params]
            serializable_data.append(f_dict)
            
        try:
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise e
