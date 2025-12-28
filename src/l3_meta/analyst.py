from __future__ import annotations

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from src.contracts import LedgerRecord
from src.shared.logger import get_logger
from src.config import config

logger = get_logger("l3.analyst")

class IndicatorPriorController:
    """
    Manages the 'Prior' distribution of indicator categories based on L2 feature importance.
    Helps the L3 Agent choose indicators that actually matter in the current regime.
    
    [V11.4 Rule] 
    prior[regime][family] = (1-α)*prior + α*normalize(importance_family)
    """
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.priors_file = self.storage_path / "indicator_priors.json"
        
        # Structure: priors[regime][category] = float (Importance weight)
        self.priors: Dict[str, Dict[str, float]] = {}
        
        # Defaults for categories (if not initialized)
        self.categories = [
            "TREND", "MOMENTUM", "VOLATILITY", "VOLUME", 
            "ADAPTIVE", "MEAN_REVERSION", "PRICE_ACTION", "PATTERN"
        ]
        
        self.alpha = 0.1  # EMA alpha (0.05~0.2 recommended)
        self.min_weight = 0.05
        self.max_weight = 0.40
        self.calibration_score = 0.5 # [V12.3] Track how well priors match reality
        
        self._load()

    def _load(self):
        if self.priors_file.exists():
            try:
                with open(self.priors_file, "r", encoding="utf-8") as f:
                    self.priors = json.load(f)
                logger.info(f"[Analyst] 인디케이터 사전 로드: {self.priors_file}")
            except Exception as e:
                logger.error(f"[Analyst] 사전 로드 실패: {e}")
        
    def _save(self):
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            with open(self.priors_file, "w", encoding="utf-8") as f:
                json.dump(self.priors, f, indent=4)
        except Exception as e:
            logger.error(f"[Analyst] 사전 저장 실패: {e}")

    def update_with_record(self, record: Any, regime_str: str, registry: Any):
        """
        Updates the prior distribution using a successful experiment record (Gate pass).
        """
        is_rejected = getattr(record, 'is_rejected', record.get('is_rejected', False) if isinstance(record, dict) else False)
        if is_rejected:
            return

        # Handle both LedgerRecord and raw metrics dict
        if isinstance(record, dict):
            trade_logic = record.get("trade_logic") or {}
        else:
            verdict = record.verdict_dump or {}
            trade_logic = verdict.get("trade_logic") or {}
            
        importance = trade_logic.get("feature_importance") or {}
        
        if not importance:
            return

        # 1. Aggregate Importance by Category
        cat_importance = {cat: 0.01 for cat in self.categories} # Small epsilon
        total_imp = 0.0
        
        for feat_name, imp_val in importance.items():
            # Feature ID might be like "VOLATILITY_ATR_V1__atr"
            # Simple heuristic: find the base feature_id before "__"
            base_id = feat_name.split("__")[0]
            meta = registry.get(base_id)
            if meta:
                cat = meta.category.upper()
                if cat in cat_importance:
                    cat_importance[cat] += imp_val
                    total_imp += imp_val

        if total_imp <= 0:
            return

        # Normalize current record's category importance
        for cat in cat_importance:
            cat_importance[cat] /= total_imp

        # 2. Update EMA Prior & [V12.3] Calibration Score
        if regime_str not in self.priors:
            self.priors[regime_str] = {cat: 1.0 / len(self.categories) for cat in self.categories}

        # Calculate alignment (simple dot product as vectors sum to 1)
        alignment_dot = 0.0
        norm_prior = 0.0
        norm_new = 0.0
        
        for cat in self.categories:
            current_p = self.priors[regime_str].get(cat, 1.0 / len(self.categories))
            new_imp = cat_importance[cat]
            
            # Dot prod
            alignment_dot += current_p * new_imp
            norm_prior += current_p ** 2
            norm_new += new_imp ** 2

            # EMA Update
            new_p = (1 - self.alpha) * current_p + self.alpha * new_imp
            
            # Clamp to prevent monopoly or extinction
            new_p = max(self.min_weight, min(self.max_weight, new_p))
            self.priors[regime_str][cat] = new_p

        # Cosine Similarity
        if norm_prior > 0 and norm_new > 0:
            cosine_sim = alignment_dot / ((norm_prior**0.5) * (norm_new**0.5))
            # Update calibration score (EMA)
            self.calibration_score = (1 - self.alpha) * self.calibration_score + self.alpha * cosine_sim
            
        # 3. Final re-normalization to ensure total = 1.0 after clamping
        s = sum(self.priors[regime_str].values())
        if s > 0:
            for cat in self.priors[regime_str]:
                self.priors[regime_str][cat] /= s

        self._save()

    def get_priors(self, regime_str: str) -> Dict[str, float]:
        """Returns the weighted distribution for categories in the given regime."""
        if regime_str not in self.priors:
            return {cat: 1.0 / len(self.categories) for cat in self.categories}
        return self.priors[regime_str]
        
    def get_calibration_score(self) -> float:
        """Returns the current calibration score (0.0 ~ 1.0)."""
        return self.calibration_score

def get_indicator_analyst(storage_path: Path) -> IndicatorPriorController:
    return IndicatorPriorController(storage_path)

