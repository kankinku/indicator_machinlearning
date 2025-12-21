
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import random

from src.l3_meta.state import RegimeState
from src.shared.logger import get_logger

logger = get_logger("meta.q_learning")

# --- RL Configuration ---
ACTIONS = [
    "TREND_FOLLOWING",  # MA, MACD, Parabolic SAR
    "MEAN_REVERSION",   # RSI, Bollinger, Stochastic
    "VOLATILITY_BREAK", # ATR, Keltner, Bands
    "MOMENTUM_ALPHA",   # ROC, CCI, AO
    "DIP_BUYING",       # Trend Long + RSI Oversold
    "DEFENSIVE"         # Strict risk, slow MAs
]

@dataclass
class QTable:
    # Key: State String, Value: List of Q-values for each action
    table: Dict[str, List[float]] = field(default_factory=dict)
    
    def get_q(self, state_key: str) -> List[float]:
        if state_key not in self.table:
            # Initialize with small variable noise to break ties, centered on 0.0
            self.table[state_key] = [random.uniform(-0.01, 0.01) for _ in ACTIONS]
        return self.table[state_key]

    def update(self, state_key: str, action_idx: int, value: float):
        if state_key not in self.table:
            self.get_q(state_key)
        self.table[state_key][action_idx] = value

class QLearner:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path / "q_table.json"
        
        # Hyperparameters (Loaded from Config)
        from src.config import config
        self.alpha = config.RL_ALPHA
        self.gamma = config.RL_GAMMA
        self.epsilon = config.RL_EPSILON_START
        self.epsilon_decay = config.RL_EPSILON_DECAY
        self.epsilon_min = config.RL_EPSILON_MIN
        
        self.q_table = QTable()
        self.last_state_key: Optional[str] = None
        self.last_action_idx: Optional[int] = None
        
        self.load()

    def get_action(self, regime: RegimeState) -> Tuple[str, int]:
        """
        Returns (ActionName, ActionIndex) based on current RegimeState.
        """
        state_key = self._encode_state(regime)
        self.last_state_key = state_key
        
        # Exploration
        if random.random() < self.epsilon:
            action_idx = random.randint(0, len(ACTIONS) - 1)
            action_name = ACTIONS[action_idx]
            logger.info(f"    [RL] Exploration (e={self.epsilon:.2f}) -> {action_name}")
            self.last_action_idx = action_idx
            return action_name, action_idx
            
        # Exploitation
        q_values = self.q_table.get_q(state_key)
        action_idx = int(np.argmax(q_values))
        action_name = ACTIONS[action_idx]
        
        logger.info(f"    [RL] Exploitation (State: {state_key}) -> {action_name} | Q-Vals: {[round(x,3) for x in q_values]}")
        
        self.last_action_idx = action_idx
        return action_name, action_idx

    def update(self, reward: float, next_regime: RegimeState):
        """
        Q(s,a) = Q(s,a) + alpha * [Reward + gamma * max(Q(s',a')) - Q(s,a)]
        """
        if self.last_state_key is None or self.last_action_idx is None:
            return

        # 1. Get Current Q(s,a)
        current_qs = self.q_table.get_q(self.last_state_key)
        current_q = current_qs[self.last_action_idx]
        
        # 2. Get Max Q(s', a')
        next_state_key = self._encode_state(next_regime)
        next_qs = self.q_table.get_q(next_state_key)
        max_next_q = max(next_qs)
        
        # 3. Compute Target
        target = reward + (self.gamma * max_next_q)
        
        # 4. Update
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table.update(self.last_state_key, self.last_action_idx, new_q)
        
        logger.info(f"    [RL] Updated Q({self.last_state_key}, {ACTIONS[self.last_action_idx]}): {current_q:.3f} -> {new_q:.3f} | Reward: {reward:.3f}")
        
        # 5. Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # 6. Save
        self.save()

    def _encode_state(self, regime: RegimeState) -> str:
        """
        Discretize continuous regime into a state key.
        Now directly uses the pre-classified 'label' from RegimeDetector.
        Format: [LABEL]
        New Labels: PANIC, GOLDILOCKS, STAGFLATION, BULL_RUN, SIDEWAYS, HIGH_VOL, BEAR_TREND
        """
        return regime.label

    def save(self):
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self.q_table.table, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save Q-Table: {e}")

    def load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    self.q_table.table = json.load(f)
                logger.info(">>> [RL] Q-Table loaded.")
            except Exception:
                logger.warning(">>> [RL] Failed to load Q-Table. Starting fresh.")
