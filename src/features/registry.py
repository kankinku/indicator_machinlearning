"""
Feature Registry - 싱글톤 패턴 적용

이 모듈은 모든 피처(지표)의 중앙 저장소입니다.
싱글톤 패턴을 사용하여 애플리케이션 전체에서 단일 인스턴스만 사용합니다.

사용법:
    from src.features.registry import get_registry
    registry = get_registry()  # 항상 동일한 인스턴스 반환
"""
import os
import json
from typing import Dict, List, Optional, TYPE_CHECKING, Type, Any
from threading import Lock

import pandas as pd
import numpy as np
import ta

from src.contracts import FeatureMetadata, TunableParamSpec, ValidationError
from src.config import config
from src.shared.logger import get_logger

if TYPE_CHECKING:
    from src.shared.interfaces import IFeatureRegistry

logger = get_logger("feature.registry")

# ============================================
# Singleton Instance Holder
# ============================================
_registry_instance: Optional["FeatureRegistry"] = None
_registry_lock = Lock()


def get_registry(registry_path: Optional[str] = None) -> "FeatureRegistry":
    """
    FeatureRegistry 싱글톤 인스턴스를 반환합니다.
    
    Args:
        registry_path: 첫 호출 시에만 사용됨. 이후 호출에서는 무시됨.
        
    Returns:
        FeatureRegistry: 싱글톤 인스턴스
        
    Example:
        >>> registry = get_registry()
        >>> feature = registry.get("RSI")
    """
    global _registry_instance
    
    if _registry_instance is None:
        with _registry_lock:
            # Double-checked locking
            if _registry_instance is None:
                # 기본 경로 사용 (config에서 가져오기)
                if registry_path is None:
                    from src.config import config
                    registry_path = str(config.FEATURE_REGISTRY_PATH)
                
                _registry_instance = FeatureRegistry(registry_path)
                _registry_instance.initialize()
                logger.info(f"[Singleton] FeatureRegistry initialized at {registry_path}")
    
    return _registry_instance


def inject_registry(registry: "FeatureRegistry") -> None:
    """
    [Performance Enhancement]
    이미 로드된 FeatureRegistry 인스턴스를 싱글톤으로 강제 주입합니다.
    멀티프로세싱 워커에서 중복 로딩을 방지하기 위해 사용됩니다.
    """
    global _registry_instance
    with _registry_lock:
        _registry_instance = registry
        logger.debug("[Singleton] FeatureRegistry instance injected.")


def reset_registry() -> None:
    """
    테스트 목적으로 싱글톤 인스턴스를 리셋합니다.
    프로덕션에서는 사용하지 마세요.
    """
    global _registry_instance
    with _registry_lock:
        _registry_instance = None
        logger.warning("[Singleton] FeatureRegistry instance reset. Use only for testing.")


class FeatureRegistry:
    """
    Central One Source of Truth for all available features (indicators).
    Manages the lifecycle of features: Registration, Retrieval, Persistence.
    
    Adheres to:
    - One Source of Truth (Single JSON file DB)
    - Singleton Pattern (Use get_registry() instead of direct instantiation)
    - Thread Safety (Locking)
    - Optimized for Multiprocessing (Pickling support, Handler Caching)
    
    Warning:
        직접 인스턴스화하지 마세요. get_registry()를 사용하세요.
    """
    
    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self._features: Dict[str, FeatureMetadata] = {}
        # Cache for compiled feature classes (e.g., {'RSI': RsiHandlerClass})
        self._handler_cache: Dict[str, Type] = {}
        self._lock = Lock()
        self._loaded = False
        
    def initialize(self):
        """Loads the registry from disk."""
        with self._lock:
            if self._loaded:
                return
            
            self._load_from_disk()
            self._loaded = True
        self._register_custom_features()

    def _register_custom_features(self) -> None:
        """
        Loads custom features at runtime without persisting them to disk.
        Keeps registry as SSOT while allowing local extensions.
        """
        try:
            from src.features.custom.loader import loader as custom_loader
            custom_loader.register_into_registry(self)
        except Exception as e:
            logger.warning(f"Custom feature registration skipped: {e}")

    def register(self, metadata: FeatureMetadata, overwrite: bool = False):
        """
        Registers a new feature into the ecosystem.
        [V2 Genome] Enforces semantic validation.
        """
        # 1. Semantic Validation
        undefined_fields = [
            f for f in ["state_logic", "transition_logic", "causality_link"]
            if getattr(metadata, f, "undefined") == "undefined"
        ]
        
        if undefined_fields:
            msg = f"[Genome v2] Feature '{metadata.feature_id}' has undefined semantics: {undefined_fields}"
            if config.STRICT_MODE:
                logger.error(f"  REJECTED: {msg}. Define State/Transition/Causality before registration.")
                raise ValueError(msg)
            else:
                logger.warning(f"  LACKING_DNA: {msg}. This feature may be retired in future phases.")

        with self._lock:
            if not overwrite and metadata.feature_id in self._features:
                raise ValueError(f"Feature {metadata.feature_id} already exists.")
            
            self._features[metadata.feature_id] = metadata
            self._save_to_disk()
            logger.info(f"Registered feature: {metadata.feature_id} ({metadata.name})")

    def register_runtime(self, metadata: FeatureMetadata, handler_cls: Type, overwrite: bool = False) -> None:
        """
        Registers a feature only for this process (no disk write).
        Intended for dynamic custom features.
        """
        with self._lock:
            if not overwrite and metadata.feature_id in self._features:
                return

            self._features[metadata.feature_id] = metadata
            self._handler_cache[metadata.feature_id] = handler_cls
            logger.info(f"Registered runtime feature: {metadata.feature_id} ({metadata.name})")

    def get(self, feature_id: str) -> Optional[FeatureMetadata]:
        if not self._loaded:
            self.initialize()
        return self._features.get(feature_id)
    
    def get_handler(self, feature_id: str) -> Optional[Type]:
        """
        Returns the compiled handler class for the feature.
        Compiles and caches it if not already done.
        """
        if feature_id in self._handler_cache:
            return self._handler_cache[feature_id]
        
        metadata = self.get(feature_id)
        if not metadata:
            return None
            
        # Compile dynamic code
        try:
            local_scope = {}
            exec(metadata.code_snippet, globals(), local_scope)
            
            if metadata.handler_func not in local_scope:
                raise ValueError(f"Handler {metadata.handler_func} not found in snippet for {feature_id}")
                
            handler_cls = local_scope[metadata.handler_func]
            
            with self._lock:
                self._handler_cache[feature_id] = handler_cls
                
            return handler_cls
        except Exception as e:
            logger.error(f"Failed to compile handler for {feature_id}: {e}")
            return None

    def warmup(self):
        """
        Pre-compiles all handlers in the registry.
        Should be called in the main process before forking/spawning workers.
        """
        if not self._loaded:
            self.initialize()
            
        logger.info("[Registry] Warming up handlers...")
        count = 0
        for feature_id in self._features:
            if self.get_handler(feature_id):
                count += 1
        logger.info(f"[Registry] Warmup complete. Compiled {count}/{len(self._features)} handlers.")

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
                
            logger.debug(f"Loaded {len(self._features)} features from registry.")
            
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

    def __getstate__(self):
        """Pickling helper: Exclude lock."""
        state = self.__dict__.copy()
        del state['_lock']
        return state

    def __setstate__(self, state):
        """Unpickling helper: Restore lock."""
        self.__dict__.update(state)
        self._lock = Lock()
