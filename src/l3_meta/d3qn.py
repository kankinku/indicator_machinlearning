"""
Dueling Double DQN - 심층 강화학습 신경망

Dueling 구조와 Double DQN을 결합한 고급 Q-Network입니다.

핵심 아이디어:
- Dueling: Value와 Advantage를 분리하여 상태 가치와 행동 이점을 독립적으로 학습
- Double: 행동 선택과 가치 평가를 분리하여 Q값 과대평가 방지

원칙:
- PyTorch 선택적 사용 (없으면 NumPy 폴백)
- GPU 가속 지원
- 모델 저장/로드 지원
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import json

from src.config import config
from src.shared.logger import get_logger

logger = get_logger("l3.d3qn")

# PyTorch 선택적 임포트
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch를 찾을 수 없습니다. NumPy 기반 폴백을 사용합니다.")


def get_device() -> str:
    """[V18] SSOT Device Selection Policy."""
    if not TORCH_AVAILABLE:
        return "cpu"
    
    if config.DEVICE_CPU_ONLY or config.DEVICE_FORCE_FALLBACK:
        return "cpu"

    mode = config.DEVICE_MODE.lower()
    
    if mode == "cpu":
        return "cpu"
    
    if mode == "cuda" and torch.cuda.is_available():
        return "cuda"
    
    if mode == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
        
    if mode == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
            
    return "cpu"


if TORCH_AVAILABLE:
    class DuelingDQN(nn.Module):
        """
        Dueling DQN 신경망 (PyTorch 구현).
        
        구조:
        ┌─────────────┐
        │   Input     │  (state_dim)
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  Shared FC  │  Linear -> ReLU -> Linear -> ReLU
        └──────┬──────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
        ┌─────────┐  ┌─────────┐
        │ Value   │  │Advantage│
        │ Stream  │  │ Stream  │
        └────┬────┘  └────┬────┘
             │            │
             └─────┬──────┘
                   │
            ┌──────▼──────┐
            │  Combine    │  Q(s,a) = V(s) + (A(s,a) - mean(A))
            └─────────────┘
        """
        
        def __init__(
            self,
            state_dim: int,
            n_actions: int,
            hidden_dim: int = None,
        ):
            """
            Args:
                state_dim: 입력 상태 차원
                n_actions: 행동 수
                hidden_dim: 은닉층 차원 (기본: config.D3QN_HIDDEN_DIM)
            """
            super().__init__()
            
            self.state_dim = state_dim
            self.n_actions = n_actions
            self.hidden_dim = hidden_dim or config.D3QN_HIDDEN_DIM
            
            # 공유 레이어
            self.shared = nn.Sequential(
                nn.Linear(state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )
            
            # Value Stream (상태 가치: V(s))
            self.value_stream = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1),
            )
            
            # Advantage Stream (행동 이점: A(s,a))
            self.advantage_stream = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, n_actions),
            )
            
            # 가중치 초기화
            self._init_weights()
        
        def _init_weights(self):
            """Xavier 초기화를 적용합니다."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """
            Q값을 계산합니다.
            
            Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
            
            Args:
                state: 상태 텐서 (batch_size, state_dim)
            
            Returns:
                Q값 텐서 (batch_size, n_actions)
            """
            shared = self.shared(state)
            
            value = self.value_stream(shared)           # (batch, 1)
            advantage = self.advantage_stream(shared)   # (batch, n_actions)
            
            # Dueling 결합: Q = V + (A - mean(A))
            q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
            
            return q_values
        
        def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
            """
            Epsilon-greedy로 행동을 선택합니다.
            
            Args:
                state: 상태 벡터
                epsilon: 탐색 확률
            
            Returns:
                선택된 행동 인덱스
            """
            if np.random.random() < epsilon:
                return np.random.randint(0, self.n_actions)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                if next(self.parameters()).is_cuda:
                    state_tensor = state_tensor.cuda()
                q_values = self.forward(state_tensor)
                return q_values.argmax(dim=-1).item()
        
        def save(self, path: Path) -> None:
            """모델을 저장합니다."""
            torch.save({
                'state_dict': self.state_dict(),
                'state_dim': self.state_dim,
                'n_actions': self.n_actions,
                'hidden_dim': self.hidden_dim,
            }, path)
            logger.info(f"모델 저장됨: {path}")
        
        @classmethod
        def load(cls, path: Path) -> 'DuelingDQN':
            """저장된 모델을 로드합니다."""
            checkpoint = torch.load(path, map_location='cpu')
            model = cls(
                state_dim=checkpoint['state_dim'],
                n_actions=checkpoint['n_actions'],
                hidden_dim=checkpoint['hidden_dim'],
            )
            model.load_state_dict(checkpoint['state_dict'])
            logger.info(f"모델 로드됨: {path}")
            return model


    class MultiHeadDuelingDQN(nn.Module):
        """
        [V11.4] Multi-head Dueling DQN.
        Shared trunk for feature extraction + Separate Advantage heads for Strategy and Risk.
        """
        def __init__(
            self,
            state_dim: int,
            head_dims: List[int],
            hidden_dim: int = None,
        ):
            super().__init__()
            self.state_dim = state_dim
            self.head_dims = head_dims
            self.hidden_dim = hidden_dim or config.D3QN_HIDDEN_DIM
            
            # Shared trunk
            self.shared = nn.Sequential(
                nn.Linear(state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )
            
            # Value Stream
            self.value_stream = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1),
            )
            
            # Advantage Heads
            self.advantage_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim // 2, out_dim),
                ) for out_dim in head_dims
            ])
            
            self._init_weights()

        def _init_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

        def forward(self, state: torch.Tensor) -> List[torch.Tensor]:
            shared = self.shared(state)
            value = self.value_stream(shared)
            
            rets = []
            for adv_head in self.advantage_heads:
                adv = adv_head(shared)
                # Combine V + (A - mean(A))
                q = value + (adv - adv.mean(dim=-1, keepdim=True))
                rets.append(q)
            return rets

        def save(self, path: Path) -> None:
            torch.save({
                'state_dict': self.state_dict(),
                'state_dim': self.state_dim,
                'head_dims': self.head_dims,
                'hidden_dim': self.hidden_dim,
            }, path)

        @classmethod
        def load(cls, path: Path) -> 'MultiHeadDuelingDQN':
            checkpoint = torch.load(path, map_location='cpu')
            model = cls(
                state_dim=checkpoint['state_dim'],
                head_dims=checkpoint['head_dims'],
                hidden_dim=checkpoint['hidden_dim'],
            )
            model.load_state_dict(checkpoint['state_dict'])
            return model

else:
    # PyTorch가 없을 때의 NumPy 폴백 구현
    class DuelingDQN:
        """
        Dueling DQN (NumPy 폴백 구현).
        
        간단한 선형 모델로 대체됩니다.
        실제 딥러닝 성능은 기대하기 어렵습니다.
        """
        
        def __init__(
            self,
            state_dim: int,
            n_actions: int,
            hidden_dim: int = None,
        ):
            self.state_dim = state_dim
            self.n_actions = n_actions
            self.hidden_dim = hidden_dim or config.D3QN_HIDDEN_DIM
            
            # 간단한 가중치 (랜덤 초기화)
            self.W = np.random.randn(state_dim, n_actions) * 0.01
            self.b = np.zeros(n_actions)
            
            logger.warning("NumPy 기반 DQN을 사용합니다. PyTorch 설치를 권장합니다.")
        
        def forward(self, state: np.ndarray) -> np.ndarray:
            """Q값을 계산합니다 (선형)."""
            if state.ndim == 1:
                state = state.reshape(1, -1)
            return state @ self.W + self.b
        
        def __call__(self, state):
            return self.forward(state)
        
        def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
            """행동을 선택합니다."""
            if np.random.random() < epsilon:
                return np.random.randint(0, self.n_actions)
            q_values = self.forward(state)
            return int(np.argmax(q_values))
        
        def parameters(self):
            """파라미터를 반환합니다 (호환성)."""
            return [self.W, self.b]
        
        def state_dict(self):
            return {'W': self.W, 'b': self.b}
        
        def load_state_dict(self, state_dict):
            self.W = state_dict['W']
            self.b = state_dict['b']
        
        def save(self, path: Path) -> None:
            """모델을 저장합니다."""
            np.savez(path, W=self.W, b=self.b, 
                    state_dim=self.state_dim, n_actions=self.n_actions)
        
        @classmethod
        def load(cls, path: Path) -> 'DuelingDQN':
            """모델을 로드합니다."""
            data = np.load(path)
            model = cls(int(data['state_dim']), int(data['n_actions']))
            model.W = data['W']
            model.b = data['b']
            return model


@dataclass
class DQNConfig:
    """DQN 설정을 담는 데이터 클래스."""
    state_dim: int
    n_actions: int
    hidden_dim: int = 256
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    
    @classmethod
    def from_config(cls, state_dim: int, n_actions: int) -> 'DQNConfig':
        """config 모듈에서 설정을 로드합니다."""
        return cls(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dim=config.D3QN_HIDDEN_DIM,
            learning_rate=config.D3QN_LEARNING_RATE,
            gamma=config.D3QN_GAMMA,
            tau=config.D3QN_TAU,
        )


def create_dqn_pair(
    state_dim: int,
    n_actions: int,
    hidden_dim: int = None,
    device: str = None,
) -> Tuple[DuelingDQN, DuelingDQN]:
    """
    Online/Target DQN 쌍을 생성합니다.
    
    Args:
        state_dim: 상태 차원
        n_actions: 행동 수
        hidden_dim: 은닉층 차원
        device: 장치 (cpu/cuda)
    
    Returns:
        (online_net, target_net) 튜플
    """
    device = device or get_device()
    hidden_dim = hidden_dim or config.D3QN_HIDDEN_DIM
    
    online_net = DuelingDQN(state_dim, n_actions, hidden_dim)
    target_net = DuelingDQN(state_dim, n_actions, hidden_dim)
    
    # Target을 Online으로 복사
    if TORCH_AVAILABLE:
        target_net.load_state_dict(online_net.state_dict())
        target_net.eval()  # Target은 학습하지 않음
        
        if device != "cpu":
            online_net = online_net.to(device)
            target_net = target_net.to(device)
    else:
        target_net.load_state_dict(online_net.state_dict())
    
    logger.info(f"DQN 쌍 생성됨 - 상태: {state_dim}, 행동: {n_actions}, 장치: {device}")
    
    return online_net, target_net


def soft_update(online_net: DuelingDQN, target_net: DuelingDQN, tau: float = None) -> None:
    """
    Target 네트워크를 Soft Update합니다.
    
    θ_target = τ * θ_online + (1 - τ) * θ_target
    
    Args:
        online_net: Online 네트워크
        target_net: Target 네트워크
        tau: 업데이트 비율 (기본: config.D3QN_TAU)
    """
    tau = tau or config.D3QN_TAU
    
    if TORCH_AVAILABLE:
        for target_param, online_param in zip(
            target_net.parameters(), 
            online_net.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1 - tau) * target_param.data
            )
    else:
        # NumPy 버전
        for key in online_net.state_dict():
            target_dict = target_net.state_dict()
            online_dict = online_net.state_dict()
            target_dict[key] = tau * online_dict[key] + (1 - tau) * target_dict[key]
        target_net.load_state_dict(target_dict)
