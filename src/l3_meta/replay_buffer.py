"""
ReplayBuffer v18 - Robust Composite Replay Buffer
- 상속(MRO) 복잡성 제거 (composition 우선)
- 단일 반환 타입 (UnifiedBatch) 보장
- Tagged/Multi-head 옵션 지원
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, NamedTuple, Union
from collections import deque
import random
import numpy as np
from threading import Lock
from src.config import config
from src.shared.logger import get_logger

logger = get_logger("l3.replay_buffer")

class Experience(NamedTuple):
    state: np.ndarray
    actions: List[int] # Single action = [idx], Multi action = [h0, h1, ...]
    reward: float
    next_state: np.ndarray
    done: bool

@dataclass
class UnifiedBatch:
    """[V18] Unifid return type for all buffer variants."""
    states: np.ndarray          # (batch, state_dim)
    actions: np.ndarray         # (batch, n_heads) - ALWAYS structure as (batch, n_heads)
    rewards: np.ndarray         # (batch,)
    next_states: np.ndarray     # (batch, state_dim)
    dones: np.ndarray           # (batch,)
    indices: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None

class ReplayBuffer:
    def __init__(
        self,
        capacity: int = None,
        batch_size: int = None,
        multi_action: bool = False,
        tagged: bool = False,
        tag_ratios: Optional[Dict[str, float]] = None,
    ):
        self.capacity = capacity or config.D3QN_BUFFER_SIZE
        self.batch_size = batch_size or config.D3QN_BATCH_SIZE
        self.multi_action = multi_action
        self.tagged = tagged
        self.tag_ratios = tag_ratios or getattr(config, "REPLAY_TAG_SAMPLE_RATIOS", {"PASS": 0.5, "NEAR_PASS": 0.3, "HARD_FAIL": 0.2})
        
        self.buffer: deque[Experience] = deque(maxlen=self.capacity)
        self.tags: deque[str] = deque(maxlen=self.capacity)
        self._lock = Lock()
        self._total_pushed = 0

    def push(self, experience: Experience, tag: str = "PASS") -> None:
        with self._lock:
            # [Step 3] Rejected Cap Logic (Optional Drop to maintain ratio)
            if self.tagged and self._should_drop_tagged(tag):
                return
            
            self.buffer.append(experience)
            if self.tagged:
                self.tags.append(tag)
            self._total_pushed += 1

    def push_transition(self, state, action_or_list, reward, next_state, done=False, tag="PASS"):
        actions = action_or_list if isinstance(action_or_list, list) else [action_or_list]
        exp = Experience(state, actions, float(reward), next_state, done)
        self.push(exp, tag=tag)

    def sample(self) -> UnifiedBatch:
        with self._lock:
            if len(self.buffer) < self.batch_size:
                raise ValueError(f"Buffer size {len(self.buffer)} < {self.batch_size}")
            
            if self.tagged and len(self.tags) == len(self.buffer):
                indices = self._sample_tagged_indices()
            else:
                indices = random.sample(range(len(self.buffer)), self.batch_size)
            
            batch = [self.buffer[i] for i in indices]
            return self._batch_to_arrays(batch)

    def _should_drop_tagged(self, tag: str) -> bool:
        if tag != "HARD_FAIL" or len(self.buffer) < config.D3QN_MIN_BUFFER_SIZE:
            return False
        
        ratios = self._get_normalized_ratios()
        target = ratios.get("HARD_FAIL", 0.0)
        if target <= 0: return True # Drop all hard fails if target 0
        
        hard_count = sum(1 for t in self.tags if t == "HARD_FAIL")
        curr_ratio = hard_count / len(self.tags) if self.tags else 0
        
        if curr_ratio > target:
            return random.random() > (target / curr_ratio)
        return False

    def _get_normalized_ratios(self) -> Dict[str, float]:
        total = sum(self.tag_ratios.values())
        if total <= 0: return {"PASS": 1.0}
        return {k: v / total for k, v in self.tag_ratios.items()}

    def _sample_tagged_indices(self) -> List[int]:
        ratios = self._get_normalized_ratios()
        tag_map: Dict[str, List[int]] = {}
        for idx, t in enumerate(self.tags):
            tag_map.setdefault(t, []).append(idx)
        
        # Allocate counts
        counts = {t: int(r * self.batch_size) for t, r in ratios.items()}
        # Ensure at least 1 if available and exists in ratio
        for t, idxs in tag_map.items():
            if counts.get(t, 0) == 0 and ratios.get(t, 0) > 0:
                counts[t] = 1
        
        # Adjust for exact batch_size
        total = sum(counts.values())
        if total != self.batch_size:
            diff = self.batch_size - total
            sorted_tags = sorted(ratios.keys(), key=lambda k: ratios[k], reverse=True)
            for t in sorted_tags:
                if diff == 0: break
                prev = counts.get(t, 0)
                if diff > 0 and len(tag_map.get(t, [])) > prev:
                    counts[t] = prev + 1
                    diff -= 1
                elif diff < 0 and prev > 0:
                    counts[t] = prev - 1
                    diff += 1

        selected = []
        pool = []
        for t, idxs in tag_map.items():
            c = min(len(idxs), counts.get(t, 0))
            if c > 0:
                selected.extend(random.sample(idxs, c))
            # Others go to pool for filling gaps
            pool.extend([i for i in idxs if i not in selected])
            
        if len(selected) < self.batch_size and pool:
            selected.extend(random.sample(pool, min(len(pool), self.batch_size - len(selected))))
            
        # Last resort fallback to uniform if still short
        if len(selected) < self.batch_size:
            rem = list(set(range(len(self.buffer))) - set(selected))
            selected.extend(random.sample(rem, self.batch_size - len(selected)))
            
        return selected

    def _batch_to_arrays(self, batch: List[Experience]) -> UnifiedBatch:
        states = np.array([e.state for e in batch], dtype=np.float32)
        # Structure actions as (batch, n_heads)
        actions = np.array([e.actions for e in batch], dtype=np.int64)
        rewards = np.array([e.reward for e in batch], dtype=np.float32)
        next_states = np.array([e.next_state for e in batch], dtype=np.float32)
        dones = np.array([e.done for e in batch], dtype=np.float32)
        
        return UnifiedBatch(states, actions, rewards, next_states, dones)

    def can_sample(self, min_size: int = None) -> bool:
        min_required = min_size or config.D3QN_MIN_BUFFER_SIZE
        return len(self.buffer) >= min_required

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def stats(self) -> dict:
        with self._lock:
            tag_counts = {}
            if self.tagged:
                for t in self.tags: tag_counts[t] = tag_counts.get(t, 0) + 1
            return {
                "size": len(self.buffer),
                "capacity": self.capacity,
                "total_pushed": self._total_pushed,
                "tag_counts": tag_counts
            }

def create_replay_buffer(multi_action=False, tagged=False, **kwargs) -> ReplayBuffer:
    return ReplayBuffer(multi_action=multi_action, tagged=tagged, **kwargs)
