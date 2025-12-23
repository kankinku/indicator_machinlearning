"""
State Encoder - 시장 상태를 신경망 입력으로 변환

이 모듈은 시장 데이터를 D3QN 신경망이 이해할 수 있는
정규화된 벡터로 변환하는 역할을 합니다.

원칙:
- 연속적인 실수값 사용 (이산 라벨 제거)
- 시계열 윈도우 적용 (추세 정보 포함)
- 정규화로 학습 안정성 확보
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json

from src.config import config
from src.shared.logger import get_logger

logger = get_logger("l3.state_encoder")


@dataclass
class EncodedState:
    """인코딩된 상태를 담는 데이터 클래스."""
    vector: np.ndarray          # (window_size * feature_dim,) 또는 (window_size, feature_dim)
    timestamp: pd.Timestamp     # 상태 생성 시점
    raw_features: Dict[str, float]  # 원본 특성값 (디버깅용)


class StateEncoder:
    """
    시장 상태 인코더.
    
    시장 데이터를 신경망 입력용 정규화된 벡터로 변환합니다.
    
    특성 구성 (12차원):
    0. VIX (변동성 지수) - 정규화됨
    1. VIX 변화율 (5일)
    2. 추세 점수 (가격 vs MA20)
    3. 모멘텀 (RSI 정규화)
    4. 수익률 (5일)
    5. 수익률 (20일)
    6. 변동성 (실현 변동성 20일)
    7. 볼린저 밴드 위치 (-1 ~ 1)
    8. 거래량 비율 (20일 평균 대비)
    9. 금리 스프레드 (10Y-2Y, 정규화)
    10. 시장 상관관계 (SPY vs QQQ)
    11. 시장 국면 점수 (연속값)
    
    시계열 윈도우:
    - 최근 N일의 특성 벡터를 스택
    - 출력: (window_size, feature_dim) 또는 flatten
    """
    
    # 특성 이름 (순서 고정)
    FEATURE_NAMES = [
        "vix_norm",            # 0. VIX 정규화
        "vix_change",          # 1. VIX 변화율
        "trend_score",         # 2. 추세 점수
        "momentum_rsi",        # 3. 모멘텀 (RSI)
        "return_5d",           # 4. 5일 수익률
        "return_20d",          # 5. 20일 수익률
        "realized_vol",        # 6. 실현 변동성
        "bb_position",         # 7. 볼린저 밴드 위치
        "volume_ratio",        # 8. 거래량 비율
        "yield_spread",        # 9. 금리 스프레드
        "market_corr",         # 10. 시장 상관관계
        "regime_score",        # 11. 시장 국면 점수
    ]
    
    def __init__(
        self,
        window_size: int = None,
        feature_dim: int = None,
        scaler_path: Optional[Path] = None,
    ):
        """
        Args:
            window_size: 시계열 윈도우 크기 (기본: config.STATE_WINDOW_SIZE)
            feature_dim: 특성 차원 (기본: config.STATE_FEATURE_DIM)
            scaler_path: 스케일러 저장 경로
        """
        self.window_size = window_size or config.STATE_WINDOW_SIZE
        self.feature_dim = feature_dim or config.STATE_FEATURE_DIM
        self.scaler_path = scaler_path or (config.LEDGER_DIR / "state_scaler.json")
        
        # 정규화 파라미터 (min-max 또는 z-score용)
        self.scaler_params: Dict[str, Dict[str, float]] = {}
        self._load_scaler()
        
        # 기본 정규화 범위 (데이터 없을 때 사용)
        self._default_ranges = {
            "vix_norm": (10, 80),          # VIX 일반 범위
            "vix_change": (-0.5, 0.5),     # VIX 일일 변화율
            "trend_score": (-1, 1),        # 추세 점수
            "momentum_rsi": (0, 100),      # RSI
            "return_5d": (-0.2, 0.2),      # 5일 수익률
            "return_20d": (-0.4, 0.4),     # 20일 수익률
            "realized_vol": (0, 0.5),      # 연환산 변동성
            "bb_position": (-2, 2),        # BB 위치 (표준편차)
            "volume_ratio": (0, 3),        # 거래량 비율
            "yield_spread": (-1, 3),       # 금리 스프레드 (%)
            "market_corr": (-1, 1),        # 상관관계
            "regime_score": (-1, 1),       # 시장 국면
        }
    
    def encode(self, df: pd.DataFrame, flatten: bool = True) -> EncodedState:
        """
        DataFrame을 상태 벡터로 변환합니다.
        
        Args:
            df: 시장 데이터 (최소 window_size + 20 행 필요)
            flatten: True면 1D 벡터, False면 2D 행렬
        
        Returns:
            EncodedState: 인코딩된 상태
        """
        if len(df) < self.window_size + 20:
            logger.warning(f"데이터 부족: {len(df)} < {self.window_size + 20}")
            # 부족한 경우 0으로 패딩
            return self._create_empty_state(df)
        
        # 열 이름 소문자 변환
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # 특성 계산
        features_df = self._compute_features(df)
        
        # 최근 window_size 행 추출
        window_features = features_df.iloc[-self.window_size:].values
        
        # 정규화
        normalized = self._normalize(window_features, features_df.columns.tolist())
        
        # NaN 처리
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 출력 형태 결정
        if flatten:
            vector = normalized.flatten()
        else:
            vector = normalized
        
        # 원본 특성 (마지막 행, 디버깅용)
        raw_features = {
            name: float(features_df[name].iloc[-1]) 
            for name in features_df.columns
        }
        
        return EncodedState(
            vector=vector,
            timestamp=df.index[-1] if hasattr(df.index, '__getitem__') else pd.Timestamp.now(),
            raw_features=raw_features,
        )
    
    def encode_from_regime(self, regime) -> np.ndarray:
        """
        RegimeState를 벡터로 변환합니다 (호환성 유지).
        
        DataFrame이 없는 경우의 폴백입니다.
        """
        # RegimeState의 연속값들을 사용
        vector = np.array([
            regime.vol_level / 50.0,        # VIX 정규화
            0.0,                            # VIX 변화율 (없음)
            regime.trend_score,             # 추세 점수
            0.5,                            # RSI (중립)
            0.0,                            # 5일 수익률
            0.0,                            # 20일 수익률
            regime.vol_level / 100.0,       # 변동성
            0.0,                            # BB 위치
            1.0,                            # 거래량 비율
            0.0,                            # 금리 스프레드
            regime.corr_score,              # 상관관계
            self._label_to_score(regime.label),  # 시장 국면
        ], dtype=np.float32)
        
        # window_size만큼 반복 (단순화)
        return np.tile(vector, self.window_size)
    
    def get_state_dim(self) -> int:
        """상태 벡터의 총 차원 수를 반환합니다."""
        return self.window_size * self.feature_dim
    
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        원시 데이터에서 특성을 계산합니다.
        """
        features = pd.DataFrame(index=df.index)
        
        close = df["close"]
        
        # 0. VIX (있는 경우)
        if "vix_close" in df.columns:
            features["vix_norm"] = df["vix_close"]
        else:
            # VIX 없으면 실현 변동성으로 대체
            features["vix_norm"] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
        
        # 1. VIX 변화율 (5일)
        features["vix_change"] = features["vix_norm"].pct_change(5)
        
        # 2. 추세 점수 (가격 vs MA20)
        ma20 = close.rolling(20).mean()
        features["trend_score"] = (close - ma20) / (ma20 + 1e-8)
        
        # 3. 모멘텀 (RSI 14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        features["momentum_rsi"] = 100 - (100 / (1 + rs))
        
        # 4. 5일 수익률
        features["return_5d"] = close.pct_change(5)
        
        # 5. 20일 수익률
        features["return_20d"] = close.pct_change(20)
        
        # 6. 실현 변동성 (20일, 연환산)
        features["realized_vol"] = close.pct_change().rolling(20).std() * np.sqrt(252)
        
        # 7. 볼린저 밴드 위치
        bb_std = close.rolling(20).std()
        features["bb_position"] = (close - ma20) / (2 * bb_std + 1e-8)
        
        # 8. 거래량 비율
        if "volume" in df.columns:
            vol_ma = df["volume"].rolling(20).mean()
            features["volume_ratio"] = df["volume"] / (vol_ma + 1e-8)
        else:
            features["volume_ratio"] = 1.0
        
        # 9. 금리 스프레드 (있는 경우)
        if "t10y2y" in df.columns:
            features["yield_spread"] = df["t10y2y"]
        else:
            features["yield_spread"] = 0.0
        
        # 10. 시장 상관관계 (SPY vs 타겟)
        if "spy_close" in df.columns:
            spy_ret = df["spy_close"].pct_change()
            target_ret = close.pct_change()
            features["market_corr"] = spy_ret.rolling(20).corr(target_ret)
        else:
            features["market_corr"] = 0.8  # 기본값
        
        # 11. 시장 국면 점수 (복합)
        # VIX 낮음 + 양 추세 + 높은 상관 = 좋은 시장
        vix_score = 1 - (features["vix_norm"] / 50).clip(0, 1)
        trend = features["trend_score"].clip(-1, 1)
        features["regime_score"] = (vix_score * 0.4 + trend * 0.4 + features["market_corr"].fillna(0) * 0.2)
        
        return features.fillna(method="ffill").fillna(0)
    
    def _normalize(self, values: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        값들을 [-1, 1] 범위로 정규화합니다.
        """
        normalized = np.zeros_like(values, dtype=np.float32)
        
        for i, name in enumerate(feature_names):
            if i >= values.shape[1]:
                break
            
            col = values[:, i]
            
            # 스케일러 파라미터 사용 또는 기본 범위
            if name in self.scaler_params:
                min_val = self.scaler_params[name]["min"]
                max_val = self.scaler_params[name]["max"]
            elif name in self._default_ranges:
                min_val, max_val = self._default_ranges[name]
            else:
                # 동적 범위 계산
                min_val, max_val = np.nanmin(col), np.nanmax(col)
                if min_val == max_val:
                    min_val, max_val = 0, 1
            
            # Min-Max 정규화 -> [-1, 1]
            normalized[:, i] = 2 * (col - min_val) / (max_val - min_val + 1e-8) - 1
        
        return np.clip(normalized, -1, 1)
    
    def _label_to_score(self, label: str) -> float:
        """시장 라벨을 연속 점수로 변환합니다."""
        label_scores = {
            "PANIC": -1.0,
            "BEAR_TREND": -0.7,
            "HIGH_VOL": -0.3,
            "STAGFLATION": -0.2,
            "SIDEWAYS": 0.0,
            "NEUTRAL": 0.0,
            "GOLDILOCKS": 0.5,
            "BULL_RUN": 1.0,
        }
        return label_scores.get(label, 0.0)
    
    def _create_empty_state(self, df: pd.DataFrame) -> EncodedState:
        """빈 상태를 생성합니다."""
        vector = np.zeros(self.get_state_dim(), dtype=np.float32)
        return EncodedState(
            vector=vector,
            timestamp=df.index[-1] if len(df) > 0 and hasattr(df.index, '__getitem__') else pd.Timestamp.now(),
            raw_features={},
        )
    
    def update_scaler(self, df: pd.DataFrame) -> None:
        """
        새 데이터로 스케일러를 업데이트합니다.
        """
        df.columns = [c.lower() for c in df.columns]
        features_df = self._compute_features(df)
        
        for name in features_df.columns:
            col = features_df[name].dropna()
            if len(col) > 0:
                self.scaler_params[name] = {
                    "min": float(col.quantile(0.01)),  # 1% 분위수 (이상치 제외)
                    "max": float(col.quantile(0.99)),  # 99% 분위수
                    "mean": float(col.mean()),
                    "std": float(col.std()),
                }
        
        self._save_scaler()
        logger.info(f"스케일러 업데이트됨: {len(self.scaler_params)} 특성")
    
    def _save_scaler(self) -> None:
        """스케일러 파라미터를 저장합니다."""
        try:
            self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.scaler_path, "w") as f:
                json.dump(self.scaler_params, f, indent=2)
        except Exception as e:
            logger.warning(f"스케일러 저장 실패: {e}")
    
    def _load_scaler(self) -> None:
        """스케일러 파라미터를 로드합니다."""
        if self.scaler_path.exists():
            try:
                with open(self.scaler_path, "r") as f:
                    self.scaler_params = json.load(f)
                logger.debug(f"스케일러 로드됨: {len(self.scaler_params)} 특성")
            except Exception as e:
                logger.warning(f"스케일러 로드 실패: {e}")


# 전역 인코더 인스턴스 (싱글톤)
_encoder_instance: Optional[StateEncoder] = None


def get_state_encoder() -> StateEncoder:
    """상태 인코더 싱글톤을 반환합니다."""
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = StateEncoder()
    return _encoder_instance


def reset_state_encoder() -> None:
    """상태 인코더를 리셋합니다 (테스트용)."""
    global _encoder_instance
    _encoder_instance = None
