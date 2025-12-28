
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import joblib
from pathlib import Path

from src.shared.logger import get_logger
from src.l2_sl.labeling.barrier import get_triple_barrier_labels, compute_daily_volatility

logger = get_logger("l2.tracker")

class TacticalModel:
    """
    L2 Tactical ML Model (The Hunter).
    Uses Feature Registry genes as inputs and Triple Barrier Labels as targets.
    """
    def __init__(self, model_type: str = "rf", params: Dict[str, Any] = None):
        self.model_type = model_type
        self.params = params or {"n_estimators": 100, "max_depth": 5, "random_state": 42}
        self.model: Optional[BaseEstimator] = None
        self._is_fitted = False
        
    def train(self, features: pd.DataFrame, close_prices: pd.Series, risk_config: Dict[str, float] = None):
        """
        Train the Tactical Model.
        
        Args:
            features (pd.DataFrame): X (The Genome Features)
            close_prices (pd.Series): Raw Close Prices for Labeling
            risk_config: {pt: 1.0, sl: 1.0, window: 10} multipliers
        """
        if risk_config is None:
            risk_config = {"pt": 1.0, "sl": 1.0, "window": 10}
            
        logger.info(f"[L2] 전술 모델 학습 시작 ({self.model_type})")
        
        # 1. Generate Labels (y)
        vol = compute_daily_volatility(close_prices)
        t_events = features.index
        
        labels = get_triple_barrier_labels(
            close_prices=close_prices,
            t_events=t_events,
            pt_sl=(risk_config["pt"], risk_config["sl"]),
            target=vol,
            max_window=int(risk_config["window"])
        )
        
        # 2. Align Data (Remove NaNs from Labeling)
        # 0 labels (TimeOut) might be treated as neutral or removed. 
        # Strategy: Train only on decisive outcomes (1: Win, -1: Loss)? 
        # Or Multi-class? For now, Binary Classification (1 vs All) for Entry signal.
        # Let's map: 1 -> 1 (Buy), -1/0 -> 0 (Hold/Ignore)
        
        y = labels.apply(lambda x: 1 if x == 1 else 0)
        
        # Valid mask
        valid_idx = ~y.isna() & ~features.isna().any(axis=1)
        X_train = features.loc[valid_idx]
        y_train = y.loc[valid_idx]
        
        if X_train.empty:
            logger.warning("[L2] 정렬 후 유효한 학습 데이터 없음")
            return
            
        # 3. Fit Model
        if self.model_type == "rf":
            self.model = RandomForestClassifier(**self.params)
        else:
            # Placeholder for LightGBM or others
            self.model = RandomForestClassifier(**self.params)
            
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        
        score = self.model.score(X_train, y_train)
        logger.info(f"[L2] 학습 완료. 정확도 {score:.4f}")
        
    def predict_uncertainty(self, features: pd.DataFrame) -> pd.Series:
        """
        Returns probability of Class 1 (Buy/Win).
        """
        if not self._is_fitted or self.model is None:
            return pd.Series(0.5, index=features.index)
            
        # Handle current step prediction (might just be 1 row)
        valid_feat = features.fillna(0) # Safety fill
        
        try:
            probs = self.model.predict_proba(valid_feat)
            # Binary classification: index 1 is positive class
            pos_probs = probs[:, 1]
            return pd.Series(pos_probs, index=features.index)
        except Exception as e:
            logger.error(f"[L2] 예측 실패: {e}")
            return pd.Series(0.5, index=features.index)
            
    def get_feature_importance(self) -> Dict[str, float]:
        if not self._is_fitted or self.model is None:
            return {}
        try:
            return dict(zip(self.model.feature_names_in_, self.model.feature_importances_))
        except:
            return {}

    def save(self, path: Path):
        if self.model:
            joblib.dump(self.model, path)
            
    def load(self, path: Path):
        if path.exists():
            self.model = joblib.load(path)
            self._is_fitted = True
