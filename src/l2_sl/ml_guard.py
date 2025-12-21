import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import joblib
import json

from src.shared.logger import get_logger
from src.l2_sl.labeling.vol_scaling import generate_triple_barrier_labels

logger = get_logger("l2.ml_guard")

class MLGuard:
    """
    Stage 3: Probabilistic ML Entry (Gatekeeper).
    Uses LightGBM to filter signals and determine position sizing based on confidence.
    """
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "class_weight": "balanced"
        }
        self.model = None
        self._is_fitted = False
        self.feature_names: List[str] = []

    def train(self, features: pd.DataFrame, targets: pd.Series = None, close_prices: pd.Series = None, risk_config: Dict[str, float] = None) -> Dict[str, float]:
        """
        Train the ML Guard model.
        
        Args:
            features (pd.DataFrame): Input features.
            targets (pd.Series): Optional pre-computed labels.
            close_prices (pd.Series): Close prices for internal labeling if targets not provided.
            risk_config (Dict): Risk parameters for automatic labeling.
        """
        if risk_config is None:
            risk_config = {"pt": 2.0, "sl": 1.0, "window": 10}
            
        logger.info("[ML Guard] Starting training...")
        
        if targets is None:
            if close_prices is None:
                logger.error("Must provide either targets or close_prices.")
                return {}
                
            # 1. Generate Triple Barrier Labels using vol_scaling
            # Labels: 1 (Profit), -1 (Loss), 0 (Invalid/Timeout)
            raw_labels_df = generate_triple_barrier_labels(
                prices=close_prices,
                k_up=risk_config.get("pt", 2.0),
                k_down=risk_config.get("sl", 1.0),
                horizon_bars=int(risk_config.get("window", 10))
            )
            raw_labels = raw_labels_df["label"]
            
            # Align features with generated labels
            common_index = features.index.intersection(raw_labels.index)
            features = features.loc[common_index]
            raw_labels = raw_labels.loc[common_index]
            y = raw_labels
        else:
            # Use provided targets
            common_index = features.index.intersection(targets.index)
            features = features.loc[common_index]
            y = targets.loc[common_index]

        self.feature_names = features.columns.tolist()
        
        # 2. Filter / Map Labels
        is_multiclass = self.params.get("objective") == "multiclass"
        
        if is_multiclass:
            # Map -1, 0, 1 -> 0, 1, 2
            # We assume labels are -1, 0, 1. If not, this might fail or need dynamic mapping.
            # Dynamic mapping:
            unique_labels = sorted(y.unique())
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
            self.inv_label_map = {i: label for label, i in self.label_map.items()}
            
            y_processed = y.map(self.label_map)
            self.params["num_class"] = len(unique_labels)
        else:
            # Binary: 1 vs Rest
            y_processed = y.apply(lambda x: 1 if x == 1 else 0)
        
        # Drop NaNs (Only labels, let LightGBM handle feature NaNs)
        valid_mask = ~y_processed.isna()
        X_train = features.loc[valid_mask]
        y_train = y_processed.loc[valid_mask]
        
        if X_train.empty:
            logger.warning("[ML Guard] No valid training data available.")
            return {}
            
        # 3. Train LightGBM
        logger.info(f"[ML Guard] Training on {len(X_train)} samples with {X_train.shape[1]} features. Objective: {self.params.get('objective', 'binary')}")
        
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        
        # 4. Evaluate (Simple In-Sample for now)
        score = self.model.score(X_train, y_train)
        logger.info(f"[ML Guard] Training completed. Accuracy: {score:.4f}")
        
        return {"accuracy": score, "n_samples": len(X_train)}

    def predict(self, features: pd.DataFrame, threshold: float = 0.7, max_prob: float = 0.9) -> pd.DataFrame:
        """
        Gatekeeper Logic: Returns Entry Decision and Position Scale.
        
        Returns:
            pd.DataFrame: Index matched to features.
                - raw_prob (float): Probability of selected class.
                - signal (int): Label (-1, 0, 1) or (0, 1) based on mapping.
                - scale (float): 0.0 to 1.0 based on confidence.
        """
        if not self._is_fitted or self.model is None:
            logger.warning("[ML Guard] Model not fitted. Returning 0 probability.")
            return pd.DataFrame({
                "raw_prob": 0.0,
                "signal": 0,
                "scale": 0.0
            }, index=features.index)
            
        # fillna
        X = features.fillna(0)
        
        try:
            probs = self.model.predict_proba(X) # (N, n_classes)
        except Exception as e:
            logger.error(f"[ML Guard] Prediction failed: {e}")
            return pd.DataFrame({
                "raw_prob": 0.0,
                "signal": 0,
                "scale": 0.0
            }, index=features.index)
            
        result = pd.DataFrame(index=features.index)
        is_multiclass = self.params.get("objective") == "multiclass"
        
        if is_multiclass:
            # probs shape: (N, n_classes)
            # Find max prob and corresponding class index
            max_probs = np.max(probs, axis=1)
            pred_indices = np.argmax(probs, axis=1)
            
            result["raw_prob"] = max_probs
            
            # Map index back to label
            # Assumption: self.inv_label_map exists
            if hasattr(self, "inv_label_map"):
                 result["params_signal"] = [self.inv_label_map.get(i, 0) for i in pred_indices]
            else:
                 result["params_signal"] = pred_indices # Fallback
            
            # Filter by threshold
            # If max_prob < threshold, force signal to 0 (Neutral)
            # UNLESS neutral (0) is the max?
            # If Model predicts 0 with high prob, it acts as filter.
            # If Model predicts 1 (Buy) with low prob (< 0.7), we treat as 0.
            
            def filter_signal(row):
                if row["raw_prob"] < threshold:
                    return 0 # Noise
                return row["params_signal"]
            
            result["signal"] = result.apply(filter_signal, axis=1)
            
        else:
            # Binary
            pos_probs = probs[:, 1]
            result["raw_prob"] = pos_probs
            result["signal"] = (pos_probs >= threshold).astype(int) # 0 or 1
            
        # Scale Logic (Shared)
        def calculate_scale(p):
            if p < threshold:
                return 0.0
            if p >= max_prob:
                return 1.0
            return (p - threshold) / (max_prob - threshold)
            
        result["scale"] = result["raw_prob"].apply(calculate_scale)
        
        return result

    def save(self, directory: Path):
        """Save model and metadata."""
        directory.mkdir(parents=True, exist_ok=True)
        if self.model:
            joblib.dump(self.model, directory / "lgb_model.joblib")
            
        # Save metadata (feature names, params)
        meta = {
            "feature_names": self.feature_names,
            "params": self.params,
            "is_fitted": self._is_fitted
        }
        with open(directory / "meta.json", "w") as f:
            json.dump(meta, f, indent=4)
        logger.info(f"[ML Guard] Model saved to {directory}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance (gain)."""
        if not self._is_fitted or self.model is None:
            return {}
        
        try:
            # Check if model has feature_importances_ (sklearn API)
            # For gain, we need to access the booster
            if hasattr(self.model, "booster_"):
                imp = self.model.booster_.feature_importance(importance_type='gain')
                return dict(zip(self.model.feature_name_, imp))
            else:
                # Fallback
                return dict(zip(self.feature_names, self.model.feature_importances_))
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

    def load(self, directory: Path):
        """Load model and metadata."""
        model_path = directory / "lgb_model.joblib"
        meta_path = directory / "meta.json"
        
        if model_path.exists():
            self.model = joblib.load(model_path)
            
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.feature_names = meta.get("feature_names", [])
                self.params = meta.get("params", {})
                self._is_fitted = meta.get("is_fitted", False)
                
        logger.info(f"[ML Guard] Model loaded from {directory}. Fitted: {self._is_fitted}")
