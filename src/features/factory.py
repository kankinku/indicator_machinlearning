
from __future__ import annotations

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import ta
from .custom.loader import loader
from src.config import config
from src.features.registry import get_registry
from src.shared.logger import get_logger

logger = get_logger("feature.factory")

# Initialize loader once
# Initialize loader once
# [Performance] Disable auto-load in workers. Use Registry as Source of Truth.
# try:
#     loader.load_all()
# except Exception as e:
#     logger.warning(f"Failed to load custom indicators: {e}")

class FeatureFactory:
    """
    V2 Feature Factory: Generates features from a Genome (Dynamic Recipe).
    
    [V11] 시장 컨텍스트 피처 강제 포함
    - RL이 선택한 피처 + 필수 컨텍스트 피처 = 최종 피처셋
    - 목적: 시장 상태 무시 전략 방지, 레짐 인식 학습 유도
    
    DI 원칙: FeatureRegistry는 싱글톤을 통해 주입받습니다.
    """
    def __init__(self, registry=None):
        # DI: 외부에서 주입받거나 싱글톤 사용
        self.registry = registry or get_registry()

    def generate_from_genome(self, df: pd.DataFrame, genome: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate features based on the provided genome using the Dynamic Registry.
        
        [V11] 컨텍스트 피처 자동 포함
        """
        if df.empty:
            return pd.DataFrame()
            
        # Standardize columns
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # [V16] Ensure no NaNs in price data before sending to indicators
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').ffill().fillna(0.0)
        
        feature_chunks = []
        
        # ============================================
        # [V11] 1. 필수 컨텍스트 피처 먼저 추가
        # ============================================
        if config.MANDATORY_CONTEXT_FEATURES:
            try:
                from src.features.context_generator import generate_context_features
                ctx_features = generate_context_features(df)
                if not ctx_features.empty:
                    feature_chunks.append(ctx_features)
                    logger.debug(f"[FeatureFactory] 컨텍스트 특징 추가: {len(ctx_features.columns)}개")
            except Exception as e:
                logger.warning(f"[FeatureFactory] 컨텍스트 특징 생성 실패: {e}")
        
        # ============================================
        # 2. RL이 선택한 Genome 피처 추가
        # ============================================
        for feature_id, params in genome.items():
            try:
                # Dynamic Generation Only
                chunk = self._try_dynamic_generation(feature_id, df, params)
                
                if chunk is not None and not chunk.empty:
                    feature_chunks.append(chunk)
                else:
                    # Log warning but continue
                    pass
                    
            except Exception as e:
                logger.warning(f"[FeatureFactory] 특징 생성 실패: {feature_id} ({e})")
                continue

        if not feature_chunks:
            return pd.DataFrame(index=df.index)
            
        features = pd.concat(feature_chunks, axis=1)
        
        # De-duplicate columns (keep first occurrence to maintain 'One Source of Truth')
        if features.columns.duplicated().any():
            dupes = features.columns[features.columns.duplicated()].unique().tolist()
            logger.warning(f"[FeatureFactory] 중복 특징 제거: {dupes}")
            features = features.loc[:, ~features.columns.duplicated()]
        
        # Handle Inf/Nan
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().fillna(0.0)
        
        return features

    def _try_dynamic_generation(self, feature_id: str, df: pd.DataFrame, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Attempts to generate feature using the Dynamic Registry.
        """
        # 1. Get pre-compiled handler from Registry
        handler_cls = self.registry.get_handler(feature_id)
        
        if not handler_cls:
            # print(f"Registry Miss or Compile Fail: {feature_id}") 
            return None
            
        # 2. Instantiate and Compute
        try:
            instance = handler_cls()
            result = instance.compute(df, **params)
            
            if result is not None and not result.empty:
                result = self._align_outputs(feature_id, result)
                # [Crucial] Prefix columns with feature_id to prevent naming collisions (e.g., LightGBM duplicates)
                result.columns = [f"{feature_id}__{c}" for c in result.columns]
                
            return result
        except Exception as e:
            import traceback
            logger.error(f"[FeatureFactory] 실행 오류: {feature_id} ({e})\n{traceback.format_exc()}")
            raise e

    def _align_outputs(self, feature_id: str, result: pd.DataFrame) -> pd.DataFrame:
        meta = self.registry.get(feature_id)
        if not meta or result is None or result.empty:
            return result

        expected_suffix = None
        if meta.outputs:
            expected_suffix = meta.outputs.get("value")

        if expected_suffix == "value" and "value" not in result.columns:
            result = result.copy()
            if result.shape[1] == 1:
                result.columns = ["value"]
            else:
                result["value"] = result.iloc[:, 0]

        return result

    def generate_single_feature(self, df: pd.DataFrame, meta: Any, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Generates a single feature for validation purposes.
        """
        # Standardize columns
        df_clean = df.copy()
        df_clean.columns = [c.lower() for c in df_clean.columns]
        
        handler_cls = self.registry.get_handler(meta.feature_id)
        if not handler_cls:
            return None
            
        instance = handler_cls()
        return instance.compute(df_clean, **params)

