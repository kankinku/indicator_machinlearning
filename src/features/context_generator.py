"""
Context Feature Generator - 시장 상태 필수 피처 생성기

[V11] 모든 전략에 강제 포함되는 시장 컨텍스트 피처를 생성합니다.

목적:
1. 시장 상태 무시 전략 방지
2. ML이 "이 시장에서는 진입 안 함" 학습 가능
3. 레짐 인식 학습 유도

생성 피처:
- CTX_VIX: 변동성 상태 (VIX 프록시)
- CTX_TREND: 추세 상태 (가격 vs MA)
- CTX_MOMENTUM: 모멘텀 상태 (ROC)
- CTX_REGIME: 복합 레짐 점수
"""
from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd

from src.config import config
from src.shared.logger import get_logger

logger = get_logger("features.context")


class ContextFeatureGenerator:
    """
    시장 컨텍스트 필수 피처 생성기.
    
    모든 전략에 강제 포함되어 ML이 시장 상태를 인식하도록 합니다.
    """
    
    def __init__(self):
        self.feature_configs = config.CONTEXT_FEATURES
        self.enabled = config.MANDATORY_CONTEXT_FEATURES
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시장 컨텍스트 피처를 생성합니다.
        
        Args:
            df: OHLCV 데이터프레임 (소문자 컬럼)
            
        Returns:
            컨텍스트 피처 데이터프레임
        """
        if not self.enabled:
            return pd.DataFrame(index=df.index)
        
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        features = pd.DataFrame(index=df.index)
        close = df["close"]
        
        # ============================================
        # 1. CTX_VIX - 변동성 상태 (정규화: 0~1)
        # ============================================
        # 실제 VIX가 있으면 사용, 없으면 실현변동성으로 대체
        if "vix_close" in df.columns or "vix" in df.columns:
            vix_col = "vix_close" if "vix_close" in df.columns else "vix"
            vix = df[vix_col]
            # VIX 정규화 (10~80 → 0~1)
            features["CTX_VIX"] = ((vix - 10) / 70).clip(0, 1)
            features["CTX_VIX_CHANGE"] = vix.pct_change(5).clip(-0.5, 0.5)
        else:
            # 실현변동성 = 20일 표준편차 × sqrt(252) (연환산)
            realized_vol = close.pct_change().rolling(20).std() * np.sqrt(252)
            # 0.1 (10%) ~ 0.5 (50%) → 0~1
            features["CTX_VIX"] = ((realized_vol - 0.1) / 0.4).clip(0, 1)
            features["CTX_VIX_CHANGE"] = realized_vol.pct_change(5).clip(-0.5, 0.5)
        
        # ============================================
        # 2. CTX_TREND - 추세 상태 (-1~1)
        # ============================================
        # 가격 vs MA20
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean() if len(close) > 200 else close.rolling(50).mean()
        
        # 복합 추세 점수
        trend_score = pd.Series(0.0, index=df.index)
        trend_score += ((close - ma20) / (ma20 + 1e-8)).clip(-0.5, 0.5) * 2  # 단기
        trend_score += ((ma20 - ma50) / (ma50 + 1e-8)).clip(-0.5, 0.5) * 2   # 중기
        trend_score += ((close - ma200) / (ma200 + 1e-8)).clip(-0.5, 0.5) * 2 # 장기
        features["CTX_TREND"] = (trend_score / 3).clip(-1, 1)
        
        # 추세 방향 (1: 상승, 0: 횡보, -1: 하락)
        features["CTX_TREND_DIR"] = np.sign(features["CTX_TREND"]).fillna(0)
        
        # ============================================
        # 3. CTX_MOMENTUM - 모멘텀 상태 (-1~1)
        # ============================================
        roc_5 = close.pct_change(5)
        roc_20 = close.pct_change(20)
        
        # 복합 모멘텀 (단기 + 중기)
        momentum = (roc_5 * 0.6 + roc_20 * 0.4)
        # -0.2 ~ 0.2 → -1 ~ 1
        features["CTX_MOMENTUM"] = (momentum / 0.2).clip(-1, 1)
        
        # RSI 기반 과매수/과매도 (0~1, 중립=0.5)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        features["CTX_RSI_LEVEL"] = (rsi / 100).clip(0, 1)
        
        # ============================================
        # 4. CTX_REGIME - 복합 레짐 점수 (-1~1)
        # ============================================
        # 좋은 시장 = 낮은 VIX + 양 추세 + 양 모멘텀
        vix_component = 1 - features["CTX_VIX"]  # VIX 낮을수록 좋음
        trend_component = features["CTX_TREND"]
        momentum_component = features["CTX_MOMENTUM"]
        
        regime_score = (
            vix_component * 0.3 +
            trend_component * 0.4 +
            momentum_component * 0.3
        )
        features["CTX_REGIME"] = regime_score.clip(-1, 1)
        
        # 레짐 카테고리 (연속값 → 이산값)
        # -1~-0.3: RISK_OFF, -0.3~0.3: NEUTRAL, 0.3~1: RISK_ON
        def categorize_regime(score):
            if score < -0.3:
                return -1  # RISK_OFF
            elif score > 0.3:
                return 1   # RISK_ON
            else:
                return 0   # NEUTRAL
        
        features["CTX_REGIME_CAT"] = features["CTX_REGIME"].apply(categorize_regime)
        
        # ============================================
        # 5. CTX_VOLATILITY - 변동성 레짐 (추가)
        # ============================================
        # 단기 vol / 장기 vol (변동성 확대/축소)
        vol_short = close.pct_change().rolling(10).std()
        vol_long = close.pct_change().rolling(50).std()
        vol_ratio = (vol_short / (vol_long + 1e-8)).clip(0.5, 2.0)
        # 0.5~2.0 → -1~1 (1보다 크면 변동성 확대, 작으면 축소)
        features["CTX_VOL_REGIME"] = ((vol_ratio - 1) / 0.5).clip(-1, 1)
        
        # ============================================
        # 6. CTX_VOLUME - 거래량 상태 (추가)
        # ============================================
        if "volume" in df.columns:
            vol_ma = df["volume"].rolling(20).mean()
            volume_ratio = df["volume"] / (vol_ma + 1e-8)
            # 0~3 → 0~1
            features["CTX_VOLUME"] = (volume_ratio / 3).clip(0, 1)
        else:
            features["CTX_VOLUME"] = 0.5  # 중립
        
        # NaN 처리
        features = features.ffill().fillna(0.0)
        
        logger.debug(f"[ContextFeatures] Generated {len(features.columns)} context features")
        
        return features
    
    def get_feature_names(self) -> list:
        """컨텍스트 피처 이름 목록을 반환합니다."""
        return [
            "CTX_VIX",
            "CTX_VIX_CHANGE",
            "CTX_TREND",
            "CTX_TREND_DIR",
            "CTX_MOMENTUM",
            "CTX_RSI_LEVEL",
            "CTX_REGIME",
            "CTX_REGIME_CAT",
            "CTX_VOL_REGIME",
            "CTX_VOLUME",
        ]


# 싱글톤 인스턴스
_context_generator: Optional[ContextFeatureGenerator] = None


def get_context_generator() -> ContextFeatureGenerator:
    """ContextFeatureGenerator 싱글톤을 반환합니다."""
    global _context_generator
    if _context_generator is None:
        _context_generator = ContextFeatureGenerator()
    return _context_generator


def generate_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """컨텍스트 피처를 생성합니다 (편의 함수)."""
    return get_context_generator().generate(df)
