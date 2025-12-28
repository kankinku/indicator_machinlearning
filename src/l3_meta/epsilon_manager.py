from typing import Dict, Optional
from src.config import config
from src.shared.logger import get_logger

logger = get_logger("l3.epsilon_manager")

class EpsilonManager:
    """
    [V15] Epsilon SSOT Manager
    Single authority for exploration rate control across all meta-agents.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EpsilonManager, cls).__new__(cls)
            # Initialize from config
            cls._instance.epsilon = config.RL_EPSILON_START
            cls._instance.decay = config.RL_EPSILON_DECAY
            cls._instance.min_eps = config.RL_EPSILON_MIN
            cls._instance.max_eps = config.RL_EPSILON_MAX
            cls._instance.step_count = 0
            cls._instance.last_reheat_step = 0
        return cls._instance

    def get_epsilon(self) -> float:
        """Standard epsilon for epsilon-greedy selection."""
        return self.epsilon

    def apply_step(self):
        """Called once per batch or experiment to decay epsilon."""
        self.step_count += 1
        if self.epsilon > self.min_eps:
            self.epsilon *= self.decay
        self.epsilon = max(self.epsilon, self.min_eps)

    def request_reheat(self, reason: str, strength: Optional[float] = None):
        """Forces epsilon back up to encourage exploration."""
        old_eps = self.epsilon
        reheat_val = strength if strength is not None else config.RL_EPSILON_REHEAT_VALUE
        self.epsilon = max(self.epsilon, reheat_val)
        self.epsilon = min(self.epsilon, self.max_eps)
        self.last_reheat_step = self.step_count
        
        logger.warning(
            f"[EpsilonManager] 리히트({reason}): "
            f"{old_eps:.3f} -> {self.epsilon:.3f}"
        )

    def reset(self, force_val: Optional[float] = None):
        """Resets epsilon to start or forced value."""
        self.epsilon = force_val if force_val is not None else config.RL_EPSILON_START
        self.step_count = 0
        self.last_reheat_step = 0
        logger.info(f"[EpsilonManager] 초기화: {self.epsilon}")

    def set_decay(self, new_decay: float):
        self.decay = new_decay

    def snapshot(self) -> Dict:
        return {
            "epsilon": round(self.epsilon, 4),
            "step_count": self.step_count,
            "last_reheat": self.last_reheat_step,
            "owner": "EpsilonManager/SSOT"
        }

def get_epsilon_manager() -> EpsilonManager:
    return EpsilonManager()
