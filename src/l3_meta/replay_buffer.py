"""
Replay Buffer - 경험 재현 버퍼

Deep RL에서 학습 안정성을 위한 경험 저장 및 샘플링 메커니즘.

원칙:
- Circular Buffer로 메모리 효율 유지
- Thread-safe 설계
- 선택적 우선순위 기반 샘플링 (PER) 지원
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, NamedTuple
from collections import deque
import random
import numpy as np
from threading import Lock

from src.config import config
from src.shared.logger import get_logger

logger = get_logger("l3.replay_buffer")


class Experience(NamedTuple):
    """
    단일 경험 (SARS' 튜플).
    
    Attributes:
        state: 현재 상태 벡터
        action: 선택한 행동 인덱스
        reward: 받은 보상
        next_state: 다음 상태 벡터
        done: 에피소드 종료 여부
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class MultiActionExperience(NamedTuple):
    """
    [V11.4] Multi-action 경험 (Integrated RL용).
    """
    state: np.ndarray
    actions: List[int]  # [head0_action, head1_action, ...]
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class BatchSample:
    """
    배치 샘플 (텐서 변환용).
    """
    states: np.ndarray          # (batch_size, state_dim)
    actions: np.ndarray         # (batch_size,)
    rewards: np.ndarray         # (batch_size,)
    next_states: np.ndarray     # (batch_size, state_dim)
    dones: np.ndarray           # (batch_size,)
    indices: Optional[np.ndarray] = None   # 우선순위 업데이트용
    weights: Optional[np.ndarray] = None   # 중요도 샘플링 가중치


@dataclass
class MultiActionBatchSample:
    """
    [V11.4] Multi-action 배치 샘플.
    """
    states: np.ndarray
    actions_list: List[np.ndarray]  # [batch_head0, batch_head1, ...]
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """
    기본 경험 재현 버퍼 (Uniform Sampling).
    
    Circular buffer 구조로 최신 경험을 유지합니다.
    """
    
    def __init__(
        self,
        capacity: int = None,
        batch_size: int = None,
    ):
        """
        Args:
            capacity: 버퍼 최대 용량 (기본: config.D3QN_BUFFER_SIZE)
            batch_size: 샘플링 배치 크기 (기본: config.D3QN_BATCH_SIZE)
        """
        self.capacity = capacity or config.D3QN_BUFFER_SIZE
        self.batch_size = batch_size or config.D3QN_BATCH_SIZE
        
        self.buffer: deque[Experience] = deque(maxlen=self.capacity)
        self._lock = Lock()
        
        # 통계
        self._total_pushed = 0
        
    def push(self, experience: Experience) -> None:
        """
        경험을 버퍼에 추가합니다.
        
        Args:
            experience: SARS' 경험 튜플
        """
        with self._lock:
            self.buffer.append(experience)
            self._total_pushed += 1
    
    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """
        개별 요소로 경험을 추가합니다 (편의 메서드).
        """
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        self.push(exp)
    
    def sample(self) -> BatchSample:
        """
        랜덤하게 배치를 샘플링합니다.
        
        Returns:
            BatchSample: 배치 데이터
        
        Raises:
            ValueError: 버퍼 크기가 배치 크기보다 작은 경우
        """
        with self._lock:
            if len(self.buffer) < self.batch_size:
                raise ValueError(
                    f"버퍼 크기({len(self.buffer)})가 배치 크기({self.batch_size})보다 작습니다"
                )
            
            batch = random.sample(list(self.buffer), self.batch_size)
        
        return self._batch_to_arrays(batch)
    
    def sample_all(self) -> BatchSample:
        """
        버퍼의 모든 경험을 반환합니다 (소규모 학습용).
        """
        with self._lock:
            batch = list(self.buffer)
        return self._batch_to_arrays(batch)
    
    def _batch_to_arrays(self, batch: List[Experience]) -> BatchSample:
        """
        경험 리스트를 numpy 배열로 변환합니다.
        """
        return BatchSample(
            states=np.array([e.state for e in batch], dtype=np.float32),
            actions=np.array([e.action for e in batch], dtype=np.int64),
            rewards=np.array([e.reward for e in batch], dtype=np.float32),
            next_states=np.array([e.next_state for e in batch], dtype=np.float32),
            dones=np.array([e.done for e in batch], dtype=np.float32),
        )
    
    def can_sample(self, min_size: int = None) -> bool:
        """
        샘플링 가능 여부를 확인합니다.
        """
        min_required = min_size or config.D3QN_MIN_BUFFER_SIZE
        return len(self.buffer) >= min_required
    
    def clear(self) -> None:
        """버퍼를 비웁니다."""
        with self._lock:
            self.buffer.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    @property
    def stats(self) -> dict:
        """버퍼 통계를 반환합니다."""
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "total_pushed": self._total_pushed,
            "utilization_pct": round(len(self.buffer) / self.capacity * 100, 2),
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    우선순위 경험 재현 버퍼 (Prioritized Experience Replay).
    
    TD 오차가 큰 경험을 더 자주 샘플링하여 학습 효율을 높입니다.
    """
    
    def __init__(
        self,
        capacity: int = None,
        batch_size: int = None,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100000,
    ):
        """
        Args:
            capacity: 버퍼 최대 용량
            batch_size: 샘플링 배치 크기
            alpha: 우선순위 지수 (0 = uniform, 1 = full priority)
            beta_start: 중요도 샘플링 보정 시작값
            beta_end: 중요도 샘플링 보정 종료값
            beta_frames: beta 증가 프레임 수
        """
        super().__init__(capacity, batch_size)
        
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        
        # 우선순위 저장 (Sum Tree 대신 간단한 리스트 사용)
        self.priorities: deque[float] = deque(maxlen=self.capacity)
        self._max_priority = 1.0
        self._frame = 0
    
    def push(self, experience: Experience) -> None:
        """우선순위와 함께 경험을 추가합니다."""
        with self._lock:
            self.buffer.append(experience)
            self.priorities.append(self._max_priority)
            self._total_pushed += 1
    
    def sample(self, frame: int = None) -> BatchSample:
        """
        우선순위 기반으로 배치를 샘플링합니다.
        
        Args:
            frame: 현재 프레임 (beta 계산용)
        """
        if frame is not None:
            self._frame = frame
        
        with self._lock:
            if len(self.buffer) < self.batch_size:
                raise ValueError("버퍼 크기가 배치 크기보다 작습니다")
            
            # 우선순위 확률 계산
            priorities = np.array(list(self.priorities), dtype=np.float32)
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            
            # 샘플링
            indices = np.random.choice(
                len(self.buffer),
                size=self.batch_size,
                replace=False,
                p=probabilities,
            )
            
            batch = [self.buffer[i] for i in indices]
            
            # 중요도 샘플링 가중치 계산
            beta = min(
                self.beta_end,
                self.beta_start + (self.beta_end - self.beta_start) * self._frame / self.beta_frames
            )
            
            weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
            weights /= weights.max()  # 정규화
        
        result = self._batch_to_arrays(batch)
        result.indices = indices
        result.weights = weights.astype(np.float32)
        
        return result
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        TD 오차 기반으로 우선순위를 업데이트합니다.
        
        Args:
            indices: 업데이트할 경험 인덱스
            priorities: 새 우선순위 값 (TD 오차 + epsilon)
        """
        with self._lock:
            for idx, priority in zip(indices, priorities):
                if 0 <= idx < len(self.priorities):
                    # deque는 인덱스 할당 지원하지 않으므로 리스트로 변환
                    priorities_list = list(self.priorities)
                    priorities_list[idx] = float(priority)
                    self.priorities = deque(priorities_list, maxlen=self.capacity)
            
            self._max_priority = max(self._max_priority, priorities.max())


class IntegratedReplayBuffer(ReplayBuffer):
    """
    [V11.4] Multi-head 전용 리플레이 버퍼 (Integrated RL용).
    """
    def push_transition(
        self,
        state: np.ndarray,
        actions: List[int],
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        exp = MultiActionExperience(
            state=state,
            actions=actions,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        self.push(exp)

    def _batch_to_arrays(self, batch: List[MultiActionExperience]) -> MultiActionBatchSample:
        """
        경험 리스트를 MultiActionBatchSample로 변환합니다.
        """
        if not batch:
            return None
            
        n_heads = len(batch[0].actions)
        actions_list = []
        for h in range(n_heads):
            actions_list.append(np.array([e.actions[h] for e in batch], dtype=np.int64))

        return MultiActionBatchSample(
            states=np.array([e.state for e in batch], dtype=np.float32),
            actions_list=actions_list,
            rewards=np.array([e.reward for e in batch], dtype=np.float32),
            next_states=np.array([e.next_state for e in batch], dtype=np.float32),
            dones=np.array([e.done for e in batch], dtype=np.float32),
        )


# 팩토리 함수
def create_replay_buffer(
    prioritized: bool = False, 
    multi_action: bool = False,
    **kwargs
) -> ReplayBuffer:
    """
    리플레이 버퍼를 생성합니다.
    
    Args:
        prioritized: True면 PER, False면 Uniform
        multi_action: True면 Integrated (Multi-Head)용 버퍼 생성
        **kwargs: 버퍼 설정
    
    Returns:
        ReplayBuffer 인스턴스
    """
    if multi_action:
        return IntegratedReplayBuffer(**kwargs)
    if prioritized:
        return PrioritizedReplayBuffer(**kwargs)
    return ReplayBuffer(**kwargs)
