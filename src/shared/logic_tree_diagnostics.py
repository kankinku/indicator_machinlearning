"""
LogicTree Diagnostics - KPI Tracking for Feature Matching
[V18] 침묵 실패(Silent Failure) 제거를 위한 진단 시스템

추적 항목:
- total_condition_evals: 조건 평가 총 횟수
- matched_direct: 직접 매칭 성공 횟수
- matched_fuzzy: prefix 매칭 성공 횟수
- ambiguous: prefix 후보 2개 이상 횟수
- unmatched: 후보 0개 횟수
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from threading import Lock
import time

from src.shared.logger import get_logger

logger = get_logger("shared.logic_tree_diagnostics")


@dataclass
class LogicTreeDiagnostics:
    """LogicTree 조건 평가 진단 결과."""
    # 카운터
    total_condition_evals: int = 0
    matched_direct: int = 0
    matched_fuzzy: int = 0
    ambiguous: int = 0
    unmatched: int = 0
    
    # 상세 기록 (디버깅용)
    unmatched_keys: List[str] = field(default_factory=list)
    ambiguous_keys: List[str] = field(default_factory=list)
    fuzzy_mappings: Dict[str, str] = field(default_factory=dict)  # {requested: actual}
    
    # 메타
    batch_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    @property
    def match_rate(self) -> float:
        """전체 매칭 성공률 (직접 + fuzzy)."""
        if self.total_condition_evals == 0:
            return 1.0  # 평가된 조건이 없으면 100%
        return (self.matched_direct + self.matched_fuzzy) / self.total_condition_evals
    
    @property
    def ambiguity_rate(self) -> float:
        """모호성 비율."""
        if self.total_condition_evals == 0:
            return 0.0
        return self.ambiguous / self.total_condition_evals
    
    @property
    def unmatched_rate(self) -> float:
        """미매칭 비율."""
        if self.total_condition_evals == 0:
            return 0.0
        return self.unmatched / self.total_condition_evals
    
    @property
    def is_healthy(self) -> bool:
        """시스템 정상 상태 여부."""
        return self.match_rate >= 0.98 and self.ambiguous == 0 and self.unmatched == 0
    
    def record_direct_match(self, feature_key: str) -> None:
        """직접 매칭 성공 기록."""
        self.total_condition_evals += 1
        self.matched_direct += 1
    
    def record_fuzzy_match(self, requested_key: str, actual_column: str) -> None:
        """Fuzzy 매칭 성공 기록."""
        self.total_condition_evals += 1
        self.matched_fuzzy += 1
        self.fuzzy_mappings[requested_key] = actual_column
    
    def record_ambiguous(self, feature_key: str, candidates: List[str]) -> None:
        """모호성(후보 2개 이상) 기록."""
        self.total_condition_evals += 1
        self.ambiguous += 1
        self.ambiguous_keys.append(f"{feature_key} -> {candidates}")
    
    def record_unmatched(self, feature_key: str) -> None:
        """미매칭(후보 0개) 기록."""
        self.total_condition_evals += 1
        self.unmatched += 1
        self.unmatched_keys.append(feature_key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary 변환 (로깅/저장용)."""
        return {
            "total_condition_evals": self.total_condition_evals,
            "matched_direct": self.matched_direct,
            "matched_fuzzy": self.matched_fuzzy,
            "ambiguous": self.ambiguous,
            "unmatched": self.unmatched,
            "match_rate": round(self.match_rate, 4),
            "ambiguity_rate": round(self.ambiguity_rate, 4),
            "unmatched_rate": round(self.unmatched_rate, 4),
            "is_healthy": self.is_healthy,
            "unmatched_keys": self.unmatched_keys[:10],  # 상위 10개만
            "ambiguous_keys": self.ambiguous_keys[:10],
            "fuzzy_mappings": dict(list(self.fuzzy_mappings.items())[:10]),
        }
    
    def log_summary(self, level: str = "info") -> None:
        """진단 결과 요약 로그 출력."""
        msg = (
            f"[LogicTree] 매칭률 {self.match_rate:.1%} | "
            f"직접 {self.matched_direct} | 유사 {self.matched_fuzzy} | "
            f"모호 {self.ambiguous} | 미매칭 {self.unmatched}"
        )
        if not self.is_healthy:
            msg += " | 상태: 비정상"
            if self.unmatched_keys:
                msg += f" | 누락: {self.unmatched_keys[:3]}"
            if self.ambiguous_keys:
                msg += f" | 모호: {self.ambiguous_keys[:3]}"
        
        if level == "warning" or not self.is_healthy:
            logger.warning(msg)
        else:
            logger.info(msg)
    
    def reset(self) -> None:
        """카운터 초기화."""
        self.total_condition_evals = 0
        self.matched_direct = 0
        self.matched_fuzzy = 0
        self.ambiguous = 0
        self.unmatched = 0
        self.unmatched_keys.clear()
        self.ambiguous_keys.clear()
        self.fuzzy_mappings.clear()
        self.timestamp = time.time()


class LogicTreeMatchError(Exception):
    """LogicTree 매칭 실패 예외 (학습 모드용)."""
    def __init__(self, message: str, feature_key: str, match_type: str):
        super().__init__(message)
        self.message = message  # Store for later access
        self.feature_key = feature_key
        self.match_type = match_type  # "ambiguous" | "unmatched"


# =========================================================
# Global Singleton for Batch-level Tracking
# =========================================================
_global_diagnostics: Optional[LogicTreeDiagnostics] = None
_lock = Lock()


def get_diagnostics() -> LogicTreeDiagnostics:
    """현재 배치의 LogicTree 진단 객체를 반환합니다."""
    global _global_diagnostics
    with _lock:
        if _global_diagnostics is None:
            _global_diagnostics = LogicTreeDiagnostics()
        return _global_diagnostics


def reset_diagnostics(batch_id: Optional[str] = None) -> LogicTreeDiagnostics:
    """새 배치 시작 시 진단 객체를 초기화합니다."""
    global _global_diagnostics
    with _lock:
        _global_diagnostics = LogicTreeDiagnostics(batch_id=batch_id)
        return _global_diagnostics


def get_and_reset_diagnostics(batch_id: Optional[str] = None) -> LogicTreeDiagnostics:
    """현재 진단 결과를 반환하고 새 배치를 위해 초기화합니다."""
    global _global_diagnostics
    with _lock:
        current = _global_diagnostics or LogicTreeDiagnostics()
        _global_diagnostics = LogicTreeDiagnostics(batch_id=batch_id)
        return current
