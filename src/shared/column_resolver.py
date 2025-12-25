"""
Column Resolver - [V18] Definitive Feature Column Resolution
"추측 매칭" 제거 및 "Feature Output Schema" 기반의 정확한 컬럼 매칭 보장.
"""
from typing import List, Optional, Set
from src.contracts import ColumnRef
from src.features.registry import get_registry
from src.shared.logger import get_logger
from src.config import config
from src.shared.logic_tree_diagnostics import get_diagnostics

logger = get_logger("shared.column_resolver")

class ColumnResolutionError(Exception):
    def __init__(self, message: str, reason: str, feature_key: str):
        super().__init__(message)
        self.reason = reason  # e.g., "FEATURE_NOT_REGISTERED", "OUTPUT_KEY_UNDEFINED", "COLUMN_NOT_FOUND"
        self.feature_key = feature_key

class ColumnResolver:
    def __init__(self, df_columns: List[str]):
        self.df_columns: Set[str] = set(df_columns)
        self.raw_columns: List[str] = list(df_columns) # For fuzzy search
        self.registry = get_registry()
        self.diag = get_diagnostics()
        
    def resolve(self, ref: ColumnRef) -> str:
        """
        ColumnRef(feature_id, output_key) -> actual_column_name
        """
        # FeatureMetadata 조회
        meta = self.registry.get(ref.feature_id)
        
        # [Case 1] Registry에 없는 Feature (동적 생성 등)
        if not meta:
            if config.LOGICTREE_STRICT:
                # 학습 모드: 엄격하게 실패 처리
                msg = f"Feature '{ref.feature_id}' not registered in FeatureRegistry."
                self.diag.record_unmatched(ref.feature_id)
                raise ColumnResolutionError(msg, "FEATURE_NOT_REGISTERED", ref.feature_id)
            else:
                # 운영 모드: Fuzzy Fallback
                return self._resolve_fuzzy(ref.feature_id, ref.output_key)

        # [Case 2] Output Key 조회
        # Default to 'value' mapping if outputs is empty (Backward Compatibility)
        outputs = meta.outputs if meta.outputs else {"value": "value"}
        
        suffix = outputs.get(ref.output_key)
        
        if not suffix:
            # 시도: 기본값 "value"로 매핑되어 있는지 확인 (legacy)
            if ref.output_key == "value":
                # meta.outputs가 비어있지 않은데 "value" 키가 없는 경우 -> 매우 이상함
                # 그래도 fuzzy로 넘기기 전에 경고
                pass
            
            if config.LOGICTREE_STRICT:
                msg = f"Output key '{ref.output_key}' not defined for feature '{ref.feature_id}'."
                self.diag.record_unmatched(ref.feature_id)
                raise ColumnResolutionError(msg, "OUTPUT_KEY_UNDEFINED", ref.feature_id)
            else:
                return self._resolve_fuzzy(ref.feature_id, ref.output_key)

        # [Case 3] Expected Column Calculation
        # Standard Rule: {feature_id}__{suffix}
        expected_col = f"{ref.feature_id}__{suffix}"
        
        # [Case 4] Existence Verification
        if expected_col in self.df_columns:
            self.diag.record_direct_match(ref.feature_id)
            return expected_col
            
        # [Case 5] Column Not Found
        if config.LOGICTREE_STRICT:
            msg = f"Resolved column '{expected_col}' for {ref.feature_id} output '{ref.output_key}' not found in DataFrame."
            self.diag.record_unmatched(ref.feature_id)
            raise ColumnResolutionError(msg, "COLUMN_NOT_FOUND", ref.feature_id)
        else:
            return self._resolve_fuzzy(ref.feature_id, ref.output_key)

    def resolve_str_key(self, feature_key: str) -> str:
        """
        Legacy support: Resolve simple string key (e.g., "RSI_V1") to column.
        Assumes output_key="value".
        """
        # Try to parse as feature_id (assuming it is one)
        ref = ColumnRef(feature_id=feature_key, output_key="value")
        return self.resolve(ref)

    def _resolve_fuzzy(self, feature_id: str, output_key: str) -> str:
        """
        1단계 Fuzzy Logic (Fallback)
        """
        prefix = f"{feature_id}__"
        candidates = [c for c in self.raw_columns if c.startswith(prefix)]
        
        if len(candidates) == 1:
            self.diag.record_fuzzy_match(feature_id, candidates[0])
            return candidates[0]
            
        if len(candidates) > 1:
            self.diag.record_ambiguous(feature_id, candidates)
            # 운영 모드 안전 선택 (알파벳순)
            selected = sorted(candidates)[0]
            logger.warning(f"[Resolver] Ambiguous '{feature_id}' -> {candidates}. Picked '{selected}'")
            return selected
            
        # Last resort: Try exact match (maybe feature_id IS the column name?)
        if feature_id in self.df_columns:
            self.diag.record_direct_match(feature_id)
            return feature_id
            
        self.diag.record_unmatched(feature_id)
        raise ColumnResolutionError(f"ColResolver failed for '{feature_id}'", "UNRESOLVED", feature_id)
