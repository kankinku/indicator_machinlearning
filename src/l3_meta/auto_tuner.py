from typing import Dict, Any, List
import numpy as np
from src.config import config
from src.shared.logger import get_logger

logger = get_logger("l3.auto_tuner")

class AutoTuner:
    """
    [V15] Self-Correcting Auto-Tuner
    Monitors system health and applies EMA-based adjustments to hyper-parameters.
    """
    def __init__(self):
        self.ema_alpha = 0.2
        self.current_weights = {
            "reward_cagr": config.SCORE_W_CAGR,
            "reward_mdd": config.SCORE_W_MDD,
            "complexity_penalty": config.SCORE_W_COMPLEXITY
        }

    def process_diagnostics(self, status: str, metrics: Dict[str, Any]):
        """
        Adjusts global scoring weights based on batch diagnostics.
        """
        if "COLLAPSED" in status:
            # Low diversity. Increase complexity penalty to force different feature sets.
            self._update_ema("complexity_penalty", self.current_weights["complexity_penalty"] * 1.2)
            
        if "STAGNANT" in status:
            # No progress. Increase reward for return to encourage aggressiveness.
            self._update_ema("reward_cagr", self.current_weights["reward_cagr"] * 1.1)
            
        if "RIGID" in status:
            # High rejection. Relax MDD penalty slightly to allow exploration.
            self._update_ema("reward_mdd", self.current_weights["reward_mdd"] * 0.9)

        # Apply to global config (Effectively updates all subsequent evaluations)
        config.SCORE_W_CAGR = self.current_weights["reward_cagr"]
        config.SCORE_W_MDD = self.current_weights["reward_mdd"]
        config.SCORE_W_COMPLEXITY = self.current_weights["complexity_penalty"]

    def _update_ema(self, key: str, target_val: float):
        old_val = self.current_weights[key]
        new_val = (1 - self.ema_alpha) * old_val + self.ema_alpha * target_val
        self.current_weights[key] = new_val
        logger.info(f"[AutoTuner] {key}: {old_val:.3f} -> {new_val:.3f}")

_tuner_instance = None

def get_auto_tuner() -> AutoTuner:
    global _tuner_instance
    if _tuner_instance is None:
        _tuner_instance = AutoTuner()
    return _tuner_instance
