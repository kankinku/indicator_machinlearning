"""
D3QN Agent - Dueling Double DQN 에이전트

기존 QLearner와 동일한 인터페이스를 제공하여 호환성을 유지합니다.
내부적으로는 신경망 기반 Deep RL을 사용합니다.

핵심 기능:
- 연속 상태 공간 처리
- 경험 재현을 통한 안정적 학습
- Double DQN으로 Q값 과대평가 방지
- Soft Update로 Target 네트워크 업데이트
"""
from __future__ import annotations

import random
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import config
from src.shared.logger import get_logger
from src.l3_meta.state import RegimeState
from src.l3_meta.state_encoder import StateEncoder, get_state_encoder
from src.l3_meta.replay_buffer import ReplayBuffer, Experience, create_replay_buffer, MultiActionExperience
from src.l3_meta.reward_shaper import RewardShaper, get_reward_shaper
from src.l3_meta.d3qn import (
    TORCH_AVAILABLE,
    get_device,
    create_dqn_pair,
    soft_update
)
from src.l3_meta.epsilon_manager import get_epsilon_manager
from src.shared.event_bus import record_event

logger = get_logger("l3.d3qn_agent")

# PyTorch 조건부 임포트
if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


# [V14] Search Profiles - High-level operational configurations
DEFAULT_ACTIONS = [
    "TREND_ALPHA",      # 2 features, tail bias (extreme structural edges)
    "TREND_STABLE",     # 3 features, center bias (stable trend patterns)
    "MOMENTUM_TAIL",    # 2 features, tail bias (overextended reversals)
    "MOMENTUM_CENTER",  # 2 features, center bias (oscillation/ranging)
    "VOLATILITY_SNIPER",# 2 features, tail bias (breakouts/shocks)
    "PATTERN_COMPLEX",  # 4 features, spread bias (multi-factor patterns)
    "DEFENSIVE_CORE",   # 2 features, center bias, (conservative trend)
    "SCALPING_FAST",    # 1-2 features, spread bias, high frequency
    "EVOLVE"            # Evolutionary RL: Elite-based generation
]


class D3QNAgent:
    """
    Dueling Double DQN 에이전트.
    
    기존 QLearner와 동일한 인터페이스를 제공합니다:
    - get_action(regime) -> (action_name, action_idx)
    - update(reward, next_regime, ...)
    - save() / load()
    
    내부적으로는 다음을 사용합니다:
    - StateEncoder: 연속 상태 인코딩
    - DuelingDQN: 신경망 기반 Q값 계산
    - ReplayBuffer: 경험 재현
    - RewardShaper: 다면적 보상 계산
    """
    
    def __init__(
        self,
        storage_path: Path,
        actions: Optional[List[str]] = None,
        model_name: str = "d3qn_model.pt",
    ):
        """
        Args:
            storage_path: 모델 저장 경로
            actions: 행동 공간 (기본: DEFAULT_ACTIONS)
            model_name: 모델 파일 이름
        """
        self.actions = actions or DEFAULT_ACTIONS
        self.n_actions = len(self.actions)
        self.storage_path = Path(storage_path)
        self.model_path = self.storage_path / model_name
        
        # 장치 설정
        self.device = get_device()
        
        # 상태 인코더
        self.state_encoder = get_state_encoder()
        self.state_dim = self.state_encoder.get_state_dim()
        
        # 신경망 (Online & Target)
        self.online_net, self.target_net = create_dqn_pair(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            device=self.device,
        )
        
        # 옵티마이저 (PyTorch 있을 때만)
        if TORCH_AVAILABLE:
            self.optimizer = torch.optim.Adam(
                self.online_net.parameters(),
                lr=config.D3QN_LEARNING_RATE,
            )
        else:
            self.optimizer = None
        
        # 경험 재현 버퍼
        self.replay_buffer = create_replay_buffer(
            prioritized=False,  # 기본은 Uniform
            tagged=getattr(config, "REPLAY_TAGGED_ENABLED", False),
            capacity=config.D3QN_BUFFER_SIZE,
            batch_size=config.D3QN_BATCH_SIZE,
        )
        
        # 보상 계산기
        self.reward_shaper = get_reward_shaper()
        
        # 하이퍼파라미터
        self.gamma = config.D3QN_GAMMA
        self.tau = config.D3QN_TAU
        self.update_freq = getattr(config, "D3QN_UPDATE_FREQ", 4)
        self.target_update_freq = getattr(config, "D3QN_TARGET_UPDATE_FREQ", 100)
        self.reheat_period = getattr(config, "RL_EPSILON_REHEAT_PERIOD", 100)
        
        # Epsilon management is now handled by get_epsilon_manager()
        self.eps_manager = get_epsilon_manager()
        
        # 상태 추적
        self.last_state: Optional[np.ndarray] = None
        self.last_action_idx: Optional[int] = None
        self.step_count = 0
        self.learn_count = 0
        
        # 학습 및 성과 통계
        self.losses: List[float] = []
        self.reward_history: List[float] = []
        self.best_reward_rolling: float = -999.0
        
        self.epsilon = config.D3QN_EPSILON # Fallback
        
        # 모델 로드 시도
        self.load()
        
        logger.info(f"D3QN 에이전트 초기화됨 - 상태: {self.state_dim}, 행동: {self.n_actions}, 장치: {self.device}")
    
    def get_action(
        self,
        regime: RegimeState,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[str, int]:
        """
        현재 상태에서 행동을 선택합니다 (epsilon-greedy).
        """
        # 상태 인코딩
        if df is not None:
            encoded = self.state_encoder.encode(df)
            state = encoded.vector
        else:
            state = self.state_encoder.encode_from_regime(regime)
        
        # Epsilon-greedy 행동 선택
        epsilon = self.eps_manager.get_epsilon()
        if random.random() < epsilon:
            action_idx = random.randint(0, self.n_actions - 1)
            exploration = True
        else:
            action_idx = self._get_best_action(state)
            exploration = False
        
        action_name = self.actions[action_idx]
        
        # 상태 저장 (update에서 사용)
        self.last_state = state
        self.last_action_idx = action_idx
        
        # 로깅
        if exploration:
            logger.debug(f"    [D3QN] 탐색 (ε={epsilon:.3f}) -> {action_name}")
        else:
            logger.debug(f"    [D3QN] 활용 -> {action_name}")
        
        return action_name, action_idx
    
    def _get_best_action(self, state: np.ndarray) -> int:
        """최적 행동을 반환합니다."""
        if TORCH_AVAILABLE:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                if self.device != "cpu":
                    state_tensor = state_tensor.to(self.device)
                q_values = self.online_net(state_tensor)
                return q_values.argmax(dim=-1).item()
        else:
            # Simple fallback for mocked linear online_net
            return int(np.argmax(self.online_net(state)))
    
    def update(
        self,
        reward: float,
        next_regime: RegimeState,
        state_key: Optional[str] = None,
        action_idx: Optional[int] = None,
        df: Optional[pd.DataFrame] = None,
        metrics: Optional[Dict] = None,
    ):
        """
        경험을 저장하고 신경망을 학습합니다.
        """
        if self.last_state is None:
            logger.warning("이전 상태가 없어 학습을 건너뜁니다")
            return
        
        action_idx = action_idx if action_idx is not None else self.last_action_idx
        if action_idx is None:
            return
        
        # 다음 상태 인코딩
        if df is not None:
            next_encoded = self.state_encoder.encode(df)
            next_state = next_encoded.vector
        else:
            next_state = self.state_encoder.encode_from_regime(next_regime)
        
        # 보상 재계산 (metrics 있으면)
        is_rejected = False
        replay_tag = "PASS"
        if metrics is not None:
            reward_breakdown = self.reward_shaper.compute_breakdown(metrics)
            reward = reward_breakdown.total
            is_rejected = reward_breakdown.is_rejected
            replay_tag = getattr(reward_breakdown, "replay_tag", "PASS")
            
        self.reward_history.append(float(reward))
        
        # [V11.2] 탈락(Rejection) 처리 - 학습을 스킵하고 싶은 경우
        if is_rejected and getattr(config, 'RL_SKIP_LEARNING_ON_REJECTION', False):
            logger.info(f"    [D3QN] Strategy REJECTED. Skipping experience storage.")
            return

        # 경험 저장
        experience = Experience(
            state=self.last_state,
            action=action_idx,
            reward=float(reward),
            next_state=next_state,
            done=False,
        )
        try:
            self.replay_buffer.push(experience, tag=replay_tag)
        except TypeError:
            self.replay_buffer.push(experience)
        
        self.step_count += 1
        
        # 학습 (일정 빈도로)
        if (
            self.step_count % self.update_freq == 0 
            and self.replay_buffer.can_sample()
        ):
            loss = self._learn()
            if loss is not None:
                self.losses.append(loss)
                self.learn_count += 1
                
                # Target 네트워크 업데이트
                if self.learn_count % self.target_update_freq == 0:
                    soft_update(self.online_net, self.target_net, self.tau)
                    logger.debug(f"    [D3QN] Target 네트워크 업데이트됨")
        
        # Stagnation check
        if self.step_count > 0 and self.step_count % self.reheat_period == 0:
            if self._is_stagnated():
                self.eps_manager.request_reheat("STAGNATION")
        
        # 주기적 저장
        if self.step_count % 100 == 0:
            self.save()

        if self.step_count % 50 == 0:
            self._log_stats()
        
        logger.info(
            f"    [D3QN] 보상: {reward:.3f} | 버퍼: {len(self.replay_buffer)} | "
            f"ε: {self.eps_manager.get_epsilon():.3f} | 학습: {self.learn_count}"
        )
        
    def _log_stats(self):
        recent_rewards = self.reward_history[-config.REWARD_STD_WINDOW:] if self.reward_history else []
        if recent_rewards:
            reward_std = float(np.std(recent_rewards))
            logger.info(f"    [D3QN] Reward Std({config.REWARD_STD_WINDOW}): {reward_std:.4f}")
            if reward_std < config.REWARD_STD_MIN:
                logger.warning(f"    [D3QN] Reward variance low: {reward_std:.6f}")
                record_event("REWARD_VARIANCE_LOW", payload={"reward_std": reward_std})
        
        try:
            stats = self.replay_buffer.stats
            tag_counts = stats.get("tag_counts", {})
            if tag_counts:
                total = sum(tag_counts.values()) or 1
                hard_ratio = tag_counts.get("HARD_FAIL", 0) / total
                near_ratio = tag_counts.get("NEAR_PASS", 0) / total
                pass_ratio = tag_counts.get("PASS", 0) / total
                logger.info("    [D3QN] Replay Mix: PASS %.2f | NEAR %.2f | HARD %.2f" % (pass_ratio, near_ratio, hard_ratio))
        except Exception:
            pass

    def _is_stagnated(self) -> bool:
        """최근 성과가 이전 기간 대비 개선되지 않았는지 확인합니다."""
        if len(self.reward_history) < self.reheat_period * 2:
            return False
            
        window = self.reheat_period
        recent_rewards = self.reward_history[-window:]
        prev_rewards = self.reward_history[-2*window:-window]
        
        recent_top_10 = np.percentile(recent_rewards, 90)
        prev_top_10 = np.percentile(prev_rewards, 90)
        
        return recent_top_10 <= prev_top_10 + 0.05

    def _learn(self) -> Optional[float]:
        if not TORCH_AVAILABLE: return None
        
        batch = self.replay_buffer.sample()
        
        states = torch.FloatTensor(batch.states).to(self.device)
        # UnifiedBatch.actions is (batch, n_heads)
        actions = torch.LongTensor(batch.actions[:, 0]).to(self.device) # Head 0 for single action
        rewards = torch.FloatTensor(batch.rewards).to(self.device)
        next_states = torch.FloatTensor(batch.next_states).to(self.device)
        dones = torch.FloatTensor(batch.dones).to(self.device)
        
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=-1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            target_q = rewards + self.gamma * next_q_values * (1 - dones)
        
        current_q = self.online_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def save(self) -> None:
        """[V18] Level-2 Persistence: Metadata JSON + Weights."""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Level 1: Metadata JSON
            meta = {
                "agent_type": "D3QNAgent",
                "epsilon": self.eps_manager.get_epsilon(),
                "step_count": self.step_count,
                "learn_count": self.learn_count,
                "actions": self.actions,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            with open(self.model_path.with_suffix('.json'), 'w') as f:
                json.dump(meta, f, indent=4)

            # Level 2: Weights
            if TORCH_AVAILABLE:
                torch.save({
                    'online_state_dict': self.online_net.state_dict(),
                    'target_state_dict': self.target_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.model_path)
            
            logger.debug(f"D3QN 모델 저장됨: {self.model_path}")
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")

    def load(self) -> None:
        """[V18] Level-2 Loading."""
        if self.model_path.exists():
            try:
                # Load Metadata first
                meta_path = self.model_path.with_suffix('.json')
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        self.step_count = meta.get('step_count', 0)
                        self.learn_count = meta.get('learn_count', 0)
                        # self.eps_manager is external SSOT, but we can log discrepancy
                
                if TORCH_AVAILABLE:
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    self.online_net.load_state_dict(checkpoint['online_state_dict'])
                    self.target_net.load_state_dict(checkpoint['target_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info(f"D3QN 모델 로드됨: {self.model_path}")
            except Exception as e:
                logger.error(f"모델 로드 실패: {e}")


class IntegratedD3QNAgent(D3QNAgent):
    """
    [V11.4] Integrated Multi-head D3QN Agent.
    학습 효율을 높이기 위해 전략(Strategy)과 리스크(Risk Profile)를 한 신경망에서 동시에 학습합니다.
    """
    def __init__(
        self,
        storage_path: Path,
        strategy_actions: List[str],
        risk_actions: List[str],
        model_name: str = "integrated_d3qn.pt"
    ):
        self.strategy_actions = strategy_actions
        self.risk_actions = risk_actions
        self.head_dims = [len(strategy_actions), len(risk_actions)]
        
        self.storage_path = Path(storage_path)
        self.model_path = self.storage_path / model_name
        
        self.device = get_device()
        self.state_encoder = get_state_encoder()
        self.state_dim = self.state_encoder.get_state_dim()
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("IntegratedD3QNAgent requires PyTorch.")

        # Multi-head Networks
        from src.l3_meta.d3qn import MultiHeadDuelingDQN
        self.online_net = MultiHeadDuelingDQN(self.state_dim, self.head_dims).to(self.device)
        self.target_net = MultiHeadDuelingDQN(self.state_dim, self.head_dims).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.D3QN_LEARNING_RATE)
        self.replay_buffer = create_replay_buffer(
            multi_action=True,
            tagged=getattr(config, "REPLAY_TAGGED_ENABLED", False),
            capacity=config.D3QN_BUFFER_SIZE,
            batch_size=config.D3QN_BATCH_SIZE,
        )
        self.reward_shaper = get_reward_shaper()
        
        # RL Hyperparams
        self.eps_manager = get_epsilon_manager()
        self.epsilon = config.D3QN_EPSILON
        self.gamma = config.D3QN_GAMMA
        self.tau = config.D3QN_TAU
        self.update_freq = getattr(config, "D3QN_UPDATE_FREQ", 4)
        
        # Monitoring
        self.step_count = 0
        self.learn_count = 0
        self.reheat_count = 0
        self.reward_history = []
        self.last_experience = None # (state, [strategy_idx, risk_idx])

        # Stagnation
        self.reheat_period = getattr(config, "D3QN_REHEAT_PERIOD", 100)
        
        self.load()

    def get_action(self, regime: RegimeState, df: Optional[pd.DataFrame] = None) -> Tuple[str, int, str, int]:
        """
        전략과 리스크 행동을 동시에 선택합니다.
        Returns: (strat_name, strat_idx, risk_name, risk_idx)
        """
        if df is not None:
            state = self.state_encoder.encode(df).vector
        else:
            state = self.state_encoder.encode_from_regime(regime)
            
        epsilon = self.eps_manager.get_epsilon()
        if random.random() < epsilon:
            s_idx = random.randint(0, self.head_dims[0] - 1)
            r_idx = random.randint(0, self.head_dims[1] - 1)
        else:
            with torch.no_grad():
                st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_heads = self.online_net(st)
                s_idx = q_heads[0].argmax(dim=-1).item()
                r_idx = q_heads[1].argmax(dim=-1).item()
        
        self.last_experience = (state, [s_idx, r_idx])
        return self.strategy_actions[s_idx], s_idx, self.risk_actions[r_idx], r_idx

    def update(
        self, 
        reward: float, 
        next_regime: RegimeState, 
        next_df: Optional[pd.DataFrame] = None,
        metrics: Optional[Dict] = None,
    ):
        if self.last_experience is None: return
        
        state, actions = self.last_experience
        self.step_count += 1
        
        if next_df is not None:
            next_state = self.state_encoder.encode(next_df).vector
        else:
            next_state = self.state_encoder.encode_from_regime(next_regime)
            
        # [V12.3] Unified Reward & Tagging
        is_rejected = False
        tag = "PASS"
        if metrics is not None:
            bd = self.reward_shaper.compute_breakdown(metrics)
            reward = bd.total
            is_rejected = bd.is_rejected
            tag = getattr(bd, "replay_tag", "PASS")
            
        self.reward_history.append(float(reward))
        
        if is_rejected and getattr(config, 'RL_SKIP_LEARNING_ON_REJECTION', False):
            self.last_experience = None
            return

        self.replay_buffer.push_transition(state, actions, float(reward), next_state, tag=tag)
        self.last_experience = None

        if (
            self.step_count % self.update_freq == 0 
            and self.replay_buffer.can_sample()
        ):
            loss = self._learn()
            if loss is not None:
                self.learn_count += 1
                if self.learn_count % 10 == 0:
                    soft_update(self.online_net, self.target_net, self.tau)

        # Reheat logic
        if self.step_count > 0 and self.step_count % self.reheat_period == 0:
            if self._is_stagnated():
                self.eps_manager.request_reheat("STAGNATION")
        
        if self.step_count % 100 == 0:
            self.save()

    def _learn(self) -> Optional[float]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch required for IntegratedD3QNAgent")
            
        batch = self.replay_buffer.sample()
        if batch is None: return None
        
        states = torch.FloatTensor(batch.states).to(self.device)
        rewards = torch.FloatTensor(batch.rewards).to(self.device)
        next_states = torch.FloatTensor(batch.next_states).to(self.device)
        dones = torch.FloatTensor(batch.dones).to(self.device)
        
        self.optimizer.zero_grad()
        
        with torch.no_grad():
            next_q_heads_online = self.online_net(next_states)
            next_q_heads_target = self.target_net(next_states)
            
        current_q_heads = self.online_net(states)
        
        total_loss = 0
        for i in range(len(self.head_dims)):
            # batch.actions is (batch, n_heads)
            head_actions = torch.LongTensor(batch.actions[:, i]).to(self.device)
            best_next_actions = next_q_heads_online[i].argmax(dim=-1)
            next_q = next_q_heads_target[i].gather(1, best_next_actions.unsqueeze(-1)).squeeze(-1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
            
            curr_q = current_q_heads[i].gather(1, head_actions.unsqueeze(-1)).squeeze(-1)
            total_loss += F.mse_loss(curr_q, target_q)
            
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()

    def save(self):
        """[V18] Integrated Save."""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            meta = {
                "agent_type": "IntegratedD3QNAgent",
                "epsilon": self.eps_manager.get_epsilon(),
                "step_count": self.step_count,
                "learn_count": self.learn_count,
                "strategy_actions": self.strategy_actions,
                "risk_actions": self.risk_actions,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            with open(self.model_path.with_suffix('.json'), 'w') as f:
                json.dump(meta, f, indent=4)

            torch.save({
                'online_state_dict': self.online_net.state_dict(),
                'target_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.model_path)
            logger.debug(f"Integrated 모델 저장됨: {self.model_path}")
        except Exception as e:
            logger.error(f"Integrated 모델 저장 실패: {e}")

    def load(self):
        """[V18] Integrated Load."""
        if self.model_path.exists():
            try:
                meta_path = self.model_path.with_suffix('.json')
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        self.step_count = meta.get('step_count', 0)
                        self.learn_count = meta.get('learn_count', 0)

                ckpt = torch.load(self.model_path, map_location=self.device)
                self.online_net.load_state_dict(ckpt['online_state_dict'])
                self.target_net.load_state_dict(ckpt.get('target_state_dict', ckpt.get('target_net_state_dict')))
                if 'optimizer_state_dict' in ckpt:
                    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                logger.info(f"Integrated D3QN loaded: {self.model_path}")
            except Exception as e:
                logger.error(f"Load failed: {e}")


def get_integrated_agent(
    storage_path: Path,
    strategy_actions: List[str],
    risk_actions: List[str]
) -> IntegratedD3QNAgent:
    return IntegratedD3QNAgent(storage_path, strategy_actions, risk_actions)


def create_rl_agent(
    storage_path: Path,
    actions: Optional[List[str]] = None,
    use_deep_rl: bool = None,
) -> 'D3QNAgent':
    use_deep = use_deep_rl if use_deep_rl is not None else config.D3QN_ENABLED
    
    if use_deep:
        if not TORCH_AVAILABLE:
            logger.warning("[D3QN] Torch unavailable. Switching to QLearner mode (Manual Fallback).")
            from src.l3_meta.q_learner import QLearner
            return QLearner(storage_path, actions)
        return D3QNAgent(storage_path, actions)
    else:
        from src.l3_meta.q_learner import QLearner
        return QLearner(storage_path, actions)
