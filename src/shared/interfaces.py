"""
인터페이스 레이어 (Dependency Inversion Principle)

이 모듈은 고수준 모듈(ledger, orchestration)과 저수준 모듈(contracts) 사이의
의존성을 역전시키기 위한 추상 인터페이스를 정의합니다.

원칙:
- 고수준 모듈은 저수준 모듈에 의존하지 않아야 함
- 둘 다 추상화에 의존해야 함
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ============================================
# 1. Feature Registry Interface
# ============================================

@runtime_checkable
class IFeatureMetadata(Protocol):
    """피처 메타데이터 인터페이스"""
    feature_id: str
    name: str
    category: str
    params: List[Any]


@runtime_checkable  
class IFeatureRegistry(Protocol):
    """피처 레지스트리 인터페이스"""
    
    def initialize(self) -> None:
        """레지스트리를 초기화합니다."""
        ...
    
    def get(self, feature_id: str) -> Optional[IFeatureMetadata]:
        """피처 ID로 메타데이터를 조회합니다."""
        ...
    
    def list_all(self) -> List[IFeatureMetadata]:
        """모든 피처를 반환합니다."""
        ...
    
    def list_by_category(self, category: str) -> List[IFeatureMetadata]:
        """카테고리별 피처를 반환합니다."""
        ...


# ============================================
# 2. Ledger Record Interface
# ============================================

@runtime_checkable
class IPolicySpec(Protocol):
    """정책 스펙 인터페이스"""
    spec_id: str
    feature_genome: Dict[str, Dict[str, Any]]
    risk_budget: Dict[str, Any]
    template_id: str


@runtime_checkable
class ILedgerRecord(Protocol):
    """레저 레코드 인터페이스"""
    exp_id: str
    timestamp: float
    policy_spec: IPolicySpec
    data_hash: str
    feature_hash: str
    label_hash: str
    cpcv_metrics: Dict[str, Any]


@runtime_checkable
class ILedgerRepository(Protocol):
    """레저 저장소 인터페이스"""
    
    def save_record(self, record: ILedgerRecord, artifact: Optional[Any] = None) -> None:
        """레코드를 저장합니다."""
        ...
    
    def load_records(self) -> List[ILedgerRecord]:
        """모든 레코드를 로드합니다."""
        ...
    
    def prune_experiments(self, keep_n: int = 100) -> int:
        """실험을 정리합니다."""
        ...


# ============================================
# 3. Model Artifact Interface
# ============================================

@runtime_checkable
class IArtifactBundle(Protocol):
    """아티팩트 번들 인터페이스"""
    
    def save(self, path: Any) -> None:
        """아티팩트를 저장합니다."""
        ...
    
    def load(self, path: Any) -> None:
        """아티팩트를 로드합니다."""
        ...


# ============================================
# 4. ML Guard Interface
# ============================================

@runtime_checkable
class IMLGuard(Protocol):
    """ML 가드 인터페이스"""
    
    def train(self, features: Any, targets: Any = None, **kwargs) -> None:
        """모델을 학습합니다."""
        ...
    
    def predict(self, features: Any, **kwargs) -> Any:
        """예측을 수행합니다."""
        ...


# ============================================
# 5. Regime State Interface
# ============================================

@dataclass
class RegimeStateDTO:
    """시장 상태 DTO (불변 데이터 전송 객체)"""
    label: str
    trend_score: float
    vol_level: float
    extra: Dict[str, Any] = field(default_factory=dict)
