# 3단계: RL/ML 고도화 구현 계획서

> 작성일: 2025-12-22
> 버전: v1.1
> 상태: ✅ 구현 완료

---

## ✅ 구현 완료 요약

**구현 일시**: 2025-12-22 14:10

**구현된 기능**:
- Dueling Double DQN (D3QN) 신경망
- 연속 상태 인코더 (12차원 × 20일 윈도우)
- 경험 재현 버퍼 (Uniform + Prioritized)
- 다면적 보상 함수 (Risk-Adjusted)
- 기존 QLearner와 호환되는 인터페이스

**테스트 결과**:
- ✅ 모든 모듈 로드 성공
- ✅ RewardShaper 정상 작동 (좋은 결과: 0.536, 나쁜 결과: -0.364)
- ✅ MetaAgent D3QN 모드 정상 작동
- ✅ 통합 테스트 통과

---

## 📋 목차

1. [현재 시스템 분석](#1-현재-시스템-분석)
2. [목표 아키텍처](#2-목표-아키텍처)
3. [구현 상세](#3-구현-상세)
4. [파일 변경 계획](#4-파일-변경-계획)
5. [테스트 계획](#5-테스트-계획)
6. [롤백 전략](#6-롤백-전략)
7. [일정](#7-일정)

---

## ✅ 체크리스트

- [x] Config에 D3QN 설정 추가
- [x] StateEncoder 구현 및 테스트
- [x] ReplayBuffer 구현 및 테스트
- [x] RewardShaper 구현 및 테스트
- [x] DuelingDQN 신경망 구현
- [x] D3QNAgent 구현 및 QLearner 인터페이스 호환
- [x] MetaAgent 통합
- [x] 통합 테스트 통과
- [x] 성능 벤치마크 확인

---

## 1. 현재 시스템 분석

### 1.1 현재 RL 구조 (Tabular Q-Learning)

```
현재 아키텍처:
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  RegimeDetector │────▶│   QLearner   │────▶│  MetaAgent  │
│  (라벨 분류)     │     │  (Q-Table)   │     │ (정책 생성)  │
└─────────────────┘     └──────────────┘     └─────────────┘
        │                      │                    │
        ▼                      ▼                    ▼
   "PANIC" 등 7개       Dict[str, List[float]]   PolicySpec
   이산 라벨             상태-행동 테이블
```

### 1.2 현재 문제점

| 문제 | 현재 상태 | 영향 |
|------|-----------|------|
| **상태 표현 단순화** | 7개 이산 라벨만 사용 (PANIC, GOLDILOCKS 등) | 세밀한 시장 상황 구분 불가 |
| **Q-Table 한계** | 상태-행동 쌍을 테이블로 저장 | 새로운 상태에 대한 일반화 불가 |
| **보상 단순화** | 평가 스코어만 보상으로 사용 | 위험 조정 수익률 미반영 |
| **exec() 사용** | 피처 코드를 동적 실행 | 보안/디버깅 어려움 |
| **시계열 무시** | 현재 시점만 고려 | 추세/모멘텀 정보 손실 |

### 1.3 현재 파일 구조

```
src/l3_meta/
├── agent.py           # MetaAgent (정책 생성)
├── q_learner.py       # Tabular Q-Learning ⬅️ 대체 대상
├── state.py           # RegimeState (상태 정의)
├── risk_profiles.py   # 리스크 프로파일
├── detectors/
│   └── regime.py      # RegimeDetector (시장 상태 분류)
└── ...
```

---

## 2. 목표 아키텍처

### 2.1 새로운 RL 구조 (Dueling Double DQN)

```
목표 아키텍처:
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  StateEncoder   │────▶│      D3QN        │────▶│   MetaAgent     │
│  (연속 상태)     │     │  (신경망 기반)    │     │  (정책 생성)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │                       │
        ▼                        ▼                       ▼
  [VIX, Trend, Vol,      ┌──────────────┐          PolicySpec
   Returns, ...]         │   V(s)       │   (Value Stream)
   실수 벡터             │   A(s,a)     │   (Advantage Stream)
   + 시계열 윈도우       └──────────────┘
                                │
                                ▼
                        ┌──────────────┐
                        │ ReplayBuffer │
                        │ (경험 재현)   │
                        └──────────────┘
```

### 2.2 핵심 개선 사항

| 영역 | 이전 | 이후 |
|------|------|------|
| **상태 표현** | 이산 라벨 (7개) | 연속 벡터 (N차원) + 시계열 윈도우 |
| **학습 알고리즘** | Tabular Q-Learning | Dueling Double DQN (D3QN) |
| **보상 함수** | 단순 스코어 | Risk-Adjusted Multi-Factor Reward |
| **경험 재현** | 없음 | Prioritized Experience Replay |
| **하드웨어** | CPU only | CPU + GPU (선택적) |

---

## 3. 구현 상세

### 3.1 상태 인코더 (StateEncoder)

**목적**: 연속적인 시장 데이터를 신경망 입력으로 변환

**파일**: `src/l3_meta/state_encoder.py` (신규)

```python
# 의사 코드
class StateEncoder:
    """
    시장 상태를 신경망 입력 벡터로 변환합니다.
    
    입력 특성 (12차원):
    - VIX (변동성 지수)
    - VIX 변화율 (5일)
    - 추세 점수 (ADX 기반)
    - 상관관계 점수 (SPY vs QQQ)
    - 최근 수익률 (5일, 20일)
    - 모멘텀 (RSI 정규화)
    - 볼린저 밴드 위치
    - 거래량 비율
    - 금리 스프레드 (10Y-2Y)
    - 달러 지수 변화율
    - 최근 변동성 (실현 변동성)
    - 시장 국면 점수 (연속값)
    
    시계열 윈도우:
    - 최근 N일 (기본 20일)의 상태 벡터를 스택
    - Shape: (window_size, feature_dim) = (20, 12)
    """
    
    def __init__(self, window_size: int = 20, feature_dim: int = 12):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.scaler = None  # 정규화기 (학습 후 저장)
        
    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """
        DataFrame을 상태 벡터로 변환합니다.
        
        Returns:
            np.ndarray: Shape (window_size, feature_dim)
        """
        pass
    
    def get_state_dim(self) -> int:
        """상태 벡터의 총 차원 수를 반환합니다."""
        return self.window_size * self.feature_dim
```

### 3.2 D3QN 신경망 (DuelingDQN)

**목적**: 상태에서 최적의 행동을 선택하는 심층 신경망

**파일**: `src/l3_meta/d3qn.py` (신규)

```python
# 의사 코드
class DuelingDQN(nn.Module):
    """
    Dueling Double DQN 신경망.
    
    구조:
    ┌─────────────┐
    │   Input     │  (window_size * feature_dim)
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Shared FC  │  Linear(state_dim, 256) + ReLU
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
    ┌─────────┐  ┌─────────┐
    │ Value   │  │Advantage│
    │ Stream  │  │ Stream  │
    │ FC(128) │  │ FC(128) │
    │ FC(1)   │  │ FC(n_a) │
    └────┬────┘  └────┬────┘
         │            │
         └─────┬──────┘
               │
        ┌──────▼──────┐
        │  Combine    │  Q(s,a) = V(s) + (A(s,a) - mean(A))
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  Output     │  (n_actions)
        └─────────────┘
    """
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        
        # 공유 레이어
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Value Stream (상태 가치)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Advantage Stream (행동 이점)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        """
        shared = self.shared(state)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        
        # Dueling 결합 공식
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values
```

### 3.3 경험 재현 버퍼 (ReplayBuffer)

**목적**: 과거 경험을 저장하고 샘플링하여 학습 안정성 향상

**파일**: `src/l3_meta/replay_buffer.py` (신규)

```python
# 의사 코드
@dataclass
class Experience:
    """단일 경험 (SARS')"""
    state: np.ndarray       # 현재 상태
    action: int             # 선택한 행동
    reward: float           # 받은 보상
    next_state: np.ndarray  # 다음 상태
    done: bool              # 종료 여부


class ReplayBuffer:
    """
    경험 재현 버퍼 (Circular Buffer).
    
    기능:
    - 최근 N개 경험 저장
    - 랜덤 미니배치 샘플링
    - 선택적: 우선순위 기반 샘플링 (PER)
    """
    
    def __init__(self, capacity: int = 10000, batch_size: int = 64):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience) -> None:
        """경험을 버퍼에 추가합니다."""
        self.buffer.append(experience)
    
    def sample(self) -> List[Experience]:
        """랜덤하게 배치를 샘플링합니다."""
        return random.sample(self.buffer, min(len(self.buffer), self.batch_size))
    
    def __len__(self) -> int:
        return len(self.buffer)
```

### 3.4 보상 엔진 (RewardShaper)

**목적**: 다면적 보상 함수로 에이전트 학습 가이드

**파일**: `src/l3_meta/reward_shaper.py` (신규)

```python
# 의사 코드
class RewardShaper:
    """
    다면적 보상 함수.
    
    보상 구성:
    R = w_return * R_return      # 수익률 보상
      + w_sharpe * R_sharpe      # 샤프 비율 보상
      + w_mdd * R_mdd            # MDD 페널티
      + w_trades * R_trades      # 거래 효율 보상
      + w_stability * R_stability # 안정성 보상
    
    각 보상 범위: [-1, 1] 정규화
    """
    
    # 보상 가중치 (config에서 로드)
    WEIGHTS = {
        "return": 0.3,       # 수익률
        "sharpe": 0.25,      # 위험 조정 수익
        "mdd": 0.2,          # 최대 낙폭 페널티
        "trades": 0.15,      # 거래 효율
        "stability": 0.1,    # 수익 안정성
    }
    
    def compute(self, metrics: Dict[str, float]) -> float:
        """
        평가 지표에서 복합 보상을 계산합니다.
        
        Args:
            metrics: {
                "total_return": float,  # 총 수익률 (%)
                "sharpe": float,        # 샤프 비율
                "mdd": float,           # 최대 낙폭 (%)
                "n_trades": int,        # 거래 횟수
                "win_rate": float,      # 승률
                "cpcv_std": float,      # 수익 변동성
            }
        
        Returns:
            float: 복합 보상 (대략 [-1, 1] 범위)
        """
        pass
    
    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """값을 [-1, 1] 범위로 정규화합니다."""
        pass
```

### 3.5 D3QN 에이전트 (D3QNAgent)

**목적**: 기존 QLearner를 대체하는 Deep RL 에이전트

**파일**: `src/l3_meta/d3qn_agent.py` (신규)

```python
# 의사 코드
class D3QNAgent:
    """
    Dueling Double DQN 에이전트.
    
    기존 QLearner와 동일한 인터페이스를 제공하여 호환성 유지.
    
    주요 메서드:
    - get_action(regime) -> (action_name, action_idx)
    - update(reward, next_regime, ...)
    - save() / load()
    
    Double DQN 로직:
    - Online Network: 행동 선택에 사용
    - Target Network: Q 값 평가에 사용
    - 주기적으로 Target을 Online으로 복사 (Soft Update)
    """
    
    def __init__(self, storage_path: Path, actions: List[str] = None):
        self.actions = actions or DEFAULT_ACTIONS
        self.n_actions = len(self.actions)
        
        # 상태 인코더
        self.state_encoder = StateEncoder()
        
        # 신경망 (Online & Target)
        state_dim = self.state_encoder.get_state_dim()
        self.online_net = DuelingDQN(state_dim, self.n_actions)
        self.target_net = DuelingDQN(state_dim, self.n_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        # 옵티마이저
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-4)
        
        # 경험 재현 버퍼
        self.replay_buffer = ReplayBuffer(capacity=10000, batch_size=64)
        
        # 보상 계산기
        self.reward_shaper = RewardShaper()
        
        # 하이퍼파라미터
        self.gamma = 0.99
        self.tau = 0.005  # Soft update 비율
        self.epsilon = 0.2
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.update_freq = 4  # 학습 빈도
        
        # 상태 추적
        self.last_state = None
        self.last_action = None
        self.step_count = 0
    
    def get_action(self, regime: RegimeState, df: pd.DataFrame = None) -> Tuple[str, int]:
        """
        현재 상태에서 행동을 선택합니다 (epsilon-greedy).
        """
        # 상태 인코딩
        state = self.state_encoder.encode(df) if df is not None else self._regime_to_vector(regime)
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.online_net(state_tensor)
                action_idx = q_values.argmax(dim=-1).item()
        
        self.last_state = state
        self.last_action = action_idx
        
        return self.actions[action_idx], action_idx
    
    def update(self, reward: float, next_regime: RegimeState, **kwargs):
        """
        경험을 저장하고 신경망을 학습합니다.
        """
        # 1. 경험 저장
        next_state = kwargs.get('next_state', self._regime_to_vector(next_regime))
        experience = Experience(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=next_state,
            done=False,
        )
        self.replay_buffer.push(experience)
        
        # 2. 학습 (일정 빈도로)
        self.step_count += 1
        if self.step_count % self.update_freq == 0 and len(self.replay_buffer) >= 64:
            self._learn()
        
        # 3. Epsilon 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _learn(self):
        """
        경험 재현 버퍼에서 샘플링하여 학습합니다.
        
        Double DQN Loss:
        - action_select = argmax(Q_online(s'))
        - Q_target = r + gamma * Q_target(s', action_select)
        - Loss = MSE(Q_online(s, a), Q_target)
        """
        batch = self.replay_buffer.sample()
        
        # 배치 텐서 변환
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.FloatTensor([e.done for e in batch])
        
        # Double DQN: Online으로 행동 선택, Target으로 가치 평가
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=-1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            target_q = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Current Q
        current_q = self.online_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Loss & Backprop
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft Update Target Network
        self._soft_update()
    
    def _soft_update(self):
        """Target 네트워크를 Soft Update합니다."""
        for target_param, online_param in zip(
            self.target_net.parameters(), 
            self.online_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1 - self.tau) * target_param.data
            )
```

---

## 4. 파일 변경 계획

### 4.1 신규 파일

| 파일 | 설명 | 우선순위 |
|------|------|----------|
| `src/l3_meta/state_encoder.py` | 상태 인코더 | 🔴 P0 |
| `src/l3_meta/d3qn.py` | Dueling DQN 신경망 | 🔴 P0 |
| `src/l3_meta/replay_buffer.py` | 경험 재현 버퍼 | 🔴 P0 |
| `src/l3_meta/reward_shaper.py` | 보상 엔진 | 🔴 P0 |
| `src/l3_meta/d3qn_agent.py` | D3QN 에이전트 | 🔴 P0 |
| `tests/unit/test_d3qn.py` | 단위 테스트 | 🟡 P1 |

### 4.2 수정 파일

| 파일 | 변경 내용 | 우선순위 |
|------|-----------|----------|
| `src/config.py` | D3QN 하이퍼파라미터 추가 | 🔴 P0 |
| `src/l3_meta/agent.py` | D3QNAgent 사용 옵션 추가 | 🔴 P0 |
| `src/l3_meta/state.py` | EncodedState 추가 | 🟡 P1 |
| `src/orchestration/infinite_loop.py` | df를 에이전트에 전달 | 🟡 P1 |

### 4.3 Config 추가 항목

```python
# D3QN 설정
D3QN_ENABLED: bool = True                    # D3QN 사용 여부 (False면 기존 Q-Table)
D3QN_HIDDEN_DIM: int = 256                   # 은닉층 차원
D3QN_LEARNING_RATE: float = 1e-4             # 학습률
D3QN_GAMMA: float = 0.99                     # 할인율
D3QN_TAU: float = 0.005                      # Soft update 비율
D3QN_BUFFER_SIZE: int = 10000                # 버퍼 크기
D3QN_BATCH_SIZE: int = 64                    # 배치 크기
D3QN_UPDATE_FREQ: int = 4                    # 학습 빈도

# 상태 인코더 설정
STATE_WINDOW_SIZE: int = 20                  # 시계열 윈도우 크기
STATE_FEATURE_DIM: int = 12                  # 특성 차원

# 보상 가중치
REWARD_W_RETURN: float = 0.30                # 수익률 가중치
REWARD_W_SHARPE: float = 0.25                # 샤프 가중치
REWARD_W_MDD: float = 0.20                   # MDD 가중치
REWARD_W_TRADES: float = 0.15                # 거래 효율 가중치
REWARD_W_STABILITY: float = 0.10             # 안정성 가중치
REWARD_MDD_THRESHOLD: float = 15.0           # MDD 페널티 임계값 (%)
```

---

## 5. 테스트 계획

### 5.1 단위 테스트

| 테스트 | 검증 항목 |
|--------|-----------|
| `test_state_encoder.py` | 입력/출력 차원, 정규화, 윈도우 처리 |
| `test_d3qn_network.py` | Forward pass, 출력 차원, Value/Advantage 분리 |
| `test_replay_buffer.py` | Push/Sample, 용량 제한, 배치 샘플링 |
| `test_reward_shaper.py` | 보상 범위, 가중치 적용, 정규화 |
| `test_d3qn_agent.py` | 행동 선택, 학습 루프, 저장/로드 |

### 5.2 통합 테스트

| 테스트 | 검증 항목 |
|--------|-----------|
| `test_d3qn_integration.py` | 전체 파이프라인 (상태→행동→보상→학습) |
| `test_backward_compatibility.py` | 기존 QLearner와 인터페이스 호환성 |

### 5.3 성능 벤치마크

| 지표 | 목표 |
|------|------|
| 학습 수렴 속도 | 기존 대비 2배 향상 |
| 최종 보상 | 기존 대비 20% 향상 |
| 메모리 사용량 | < 2GB |
| GPU 사용량 (선택적) | < 50% |

---

## 6. 롤백 전략

### 6.1 Feature Flag

```python
# config.py
D3QN_ENABLED: bool = True  # False로 설정하면 기존 QLearner 사용
```

### 6.2 인터페이스 호환성

```python
# agent.py에서
if config.D3QN_ENABLED:
    self.strategy_rl = D3QNAgent(repo.base_dir)
else:
    self.strategy_rl = QLearner(repo.base_dir)  # 기존 방식
```

### 6.3 저장된 모델 마이그레이션

```python
# 기존 Q-Table이 있으면 D3QN 초기화에 활용
def migrate_q_table_to_d3qn(q_table_path: Path, d3qn_agent: D3QNAgent):
    """Q-Table의 지식을 D3QN 사전 학습에 활용합니다."""
    pass
```

---

## 7. 일정

### Phase 3.1: 기반 구조 (예상 소요: 1시간)

1. ✅ Config에 D3QN 설정 추가
2. ✅ StateEncoder 구현
3. ✅ ReplayBuffer 구현
4. ✅ RewardShaper 구현

### Phase 3.2: 신경망 구현 (예상 소요: 1시간)

1. ✅ DuelingDQN 신경망 구현
2. ✅ D3QNAgent 구현
3. ✅ 기존 인터페이스 호환성 확보

### Phase 3.3: 통합 (예상 소요: 30분)

1. ✅ MetaAgent에 D3QN 옵션 통합
2. ✅ infinite_loop에 df 전달 추가
3. ✅ Feature Flag 동작 확인

### Phase 3.4: 테스트 (예상 소요: 30분)

1. ✅ 단위 테스트 작성
2. ✅ 통합 테스트 실행
3. ✅ 성능 벤치마크

---

## 📐 의존성

### 필수 패키지

```txt
torch>=2.0.0           # PyTorch (D3QN 신경망)
numpy>=1.24.0          # 수치 연산
pandas>=2.0.0          # 데이터 처리 (이미 설치됨)
```

### 선택 패키지

```txt
tensorboard>=2.15.0    # 학습 시각화 (선택)
```

---

## 🔍 참고 자료

- [Dueling DQN 논문](https://arxiv.org/abs/1511.06581)
- [Double DQN 논문](https://arxiv.org/abs/1509.06461)
- [금융 RL 적용 사례](https://arxiv.org/abs/2111.05188)

---

## ✅ 체크리스트

- [ ] Config에 D3QN 설정 추가
- [ ] StateEncoder 구현 및 테스트
- [ ] ReplayBuffer 구현 및 테스트
- [ ] RewardShaper 구현 및 테스트
- [ ] DuelingDQN 신경망 구현
- [ ] D3QNAgent 구현 및 QLearner 인터페이스 호환
- [ ] MetaAgent 통합
- [ ] 통합 테스트 통과
- [ ] 성능 벤치마크 확인
