
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from dataclasses import dataclass, field
import random
import json
from pathlib import Path

from src.shared.logger import get_logger

logger = get_logger("l1.manager")

# Actions for Operational Manager
# 0: HOLD (Keep current position)
# 1: EXIT_50 (Sell half)
# 2: EXIT_100 (Sell all)
# 3: TRAIL_STOP (Tighten stop loss by 20%)
ACTIONS = ["HOLD", "EXIT_50", "EXIT_100", "TRAIL_STOP"]

@dataclass
class PositionState:
    entry_price: float
    current_price: float
    size: float
    duration: int
    confidence: float # L2 Model Score
    atr: float
    stop_loss: float
    
    @property
    def pnl_pct(self) -> float:
        return (self.current_price - self.entry_price) / self.entry_price

class OprManagerRL:
    """
    L1 Operational Agent (The Manager).
    Manages an open position using RL to optimize exit and risk.
    """
    def __init__(self, storage_path: Path):
        self.q_table: Dict[str, np.ndarray] = {} # State Hash -> Q-Values
        self.storage_path = storage_path / "l1_q_table.json"
        
        # Hyperparameters
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.1
        
        self.load()

    def decide_action(self, state: PositionState) -> Tuple[str, int]:
        """
        Decide operational action based on position state.
        """
        state_key = self._encode_state(state)
        
        # Initialize Q-values if new state
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(ACTIONS))
            
        # Epsilon-Greedy
        if random.random() < self.epsilon:
            action_idx = random.randint(0, len(ACTIONS) - 1)
        else:
            action_idx = int(np.argmax(self.q_table[state_key]))
            
        return ACTIONS[action_idx], action_idx

    def learn(self, state: PositionState, action_idx: int, reward: float, next_state: PositionState, done: bool):
        """
        Update different Q-value based on reward.
        """
        state_key = self._encode_state(state)
        current_q = self.q_table[state_key][action_idx]
        
        if done:
            target = reward
        else:
            next_key = self._encode_state(next_state)
            if next_key not in self.q_table:
                self.q_table[next_key] = np.zeros(len(ACTIONS))
            max_next_q = np.max(self.q_table[next_key])
            target = reward + self.gamma * max_next_q
            
        # Update
        self.q_table[state_key][action_idx] = current_q + self.alpha * (target - current_q)

    def _encode_state(self, s: PositionState) -> str:
        """
        Discretize continuous state into buckets for Q-Table.
        State: [PnL, Duration, Confidence, Volatility]
        """
        # PnL Buckets: < -2%, -2~0%, 0~2%, 2~5%, > 5%
        if s.pnl_pct < -0.02: pnl_b = "LOSS_BIG"
        elif s.pnl_pct < 0: pnl_b = "LOSS_SMALL"
        elif s.pnl_pct < 0.02: pnl_b = "PROFIT_SMALL"
        elif s.pnl_pct < 0.05: pnl_b = "PROFIT_MED"
        else: pnl_b = "PROFIT_BIG"
        
        # Duration Buckets: Short (<5), Med (5-20), Long (>20)
        if s.duration < 5: dur_b = "SHORT"
        elif s.duration < 20: dur_b = "MED"
        else: dur_b = "LONG"
        
        # Confidence: Low (<0.6), High (>=0.6)
        conf_b = "HIGH" if s.confidence >= 0.6 else "LOW"
        
        return f"{pnl_b}_{dur_b}_{conf_b}"

    def save(self):
        # Convert np array to list for JSON
        serializable = {k: v.tolist() for k, v in self.q_table.items()}
        try:
            with open(self.storage_path, "w") as f:
                json.dump(serializable, f)
        except Exception as e:
            logger.error(f"[L1] Q-Table 저장 실패: {e}")

    def load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    self.q_table = {k: np.array(v) for k, v in data.items()}
            except Exception:
                pass
                
