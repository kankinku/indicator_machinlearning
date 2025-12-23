# 🔍 전략 학습 시스템 전체 분석 보고서

## 📊 시스템 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          L3 META (Director)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │ RegimeDetector│→│ D3QN/Q-Table │→│ MetaAgent (Policy Generator) │  │
│  │   (State)    │  │   (Action)   │  │  - Feature Selection         │  │
│  └──────────────┘  └──────────────┘  │  - Parameter Evolution        │  │
│                                       │  - Risk Budget Sampling       │  │
│                                       └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓ PolicySpec
┌─────────────────────────────────────────────────────────────────────────┐
│                          L2 Supervised Learning                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │FeatureFactory│→│TripleBarrier │→│    MLGuard (LightGBM)        │  │
│  │ (Genome→피처) │  │   Labels    │  │  - Train on Labels           │  │
│  └──────────────┘  └──────────────┘  │  - Predict Probabilities     │  │
│                                       └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓ Signal (pred, prob, scale)
┌─────────────────────────────────────────────────────────────────────────┐
│                          L1 Backtest & Validation                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │SignalBacktest│→│ CPCV/PBO     │→│    Validation (Hard Gates)   │  │
│  │ (실거래시뮬)  │  │   계산      │  │  - Trade Count, MDD, WinRate │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓ Metrics
┌─────────────────────────────────────────────────────────────────────────┐
│                          Reward & Evolution                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │ RewardShaper │→│ Curriculum   │→│     RL Update (Q-Value)      │  │
│  │  (Scoring)   │  │  Controller  │  │  - Q(s,a) ← reward + γ·Q'   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ L3: RL 에이전트의 지표/파라미터 선택

### 현재 구현

**State (상태 인코딩)**
```
RegimeDetector → RegimeState(label, trend_score, vol_level, corr_score)
                         ↓
StateEncoder → 12차원 연속 벡터 × 20 윈도우 = 240차원
```

**Actions (행동 공간)**
```python
DEFAULT_ACTIONS = [
    "TREND_FOLLOWING",   # MA, MACD
    "MEAN_REVERSION",    # RSI, Bollinger
    "VOLATILITY_BREAK",  # ATR, Keltner
    "MOMENTUM_ALPHA",    # ROC, CCI
    "DIP_BUYING",        # Trend + RSI Oversold
    "DEFENSIVE"          # Strict risk, slow MAs
]
```

**Flow**:
1. `RegimeDetector` → 시장 상태 분류 (PANIC, BULL_RUN, SIDEWAYS 등)
2. `D3QN/QLearner.get_action(regime)` → 전략 유형 선택 (ε-greedy)
3. `MetaAgent._construct_genome_from_action()` → 지표 선택
   - Action → Category 매핑 (TREND_FOLLOWING → ["TREND"])
   - Registry에서 해당 Category 지표 랜덤 선택
4. `MetaAgent._evolve_params()` → 파라미터 진화
   - Config 범위 내에서 랜덤 샘플링
   - Regime/Strategy 기반 조정 (PANIC → 짧은 window)
5. `MetaAgent._make_spec()` → Risk Budget 샘플링
   - k_up, k_down, horizon 랜덤 샘플링

### 🔴 문제점

| 문제 | 설명 | 영향도 |
|------|------|--------|
| **Action 추상화 수준** | "TREND_FOLLOWING"은 너무 고수준. 실제 지표 선택은 랜덤 | 높음 |
| **지표 선택 무작위성** | Category 내에서 완전 랜덤 → RL이 학습할 가치 없음 | 심각 |
| **파라미터 탐색 비효율** | 랜덤 샘플링 → 좋은 조합 발견 확률 낮음 | 중간 |
| **행동 공간 비표현력** | 6개 전략 유형 → 지표 200개+ 조합 표현 불가 | 높음 |

---

## 2️⃣ L2: ML Guard (LightGBM) 학습

### 현재 구현

**라벨링 (Triple Barrier)**
```python
generate_triple_barrier_labels(
    prices=close,
    k_up=k_up,      # TP 배수 (vol 기준)
    k_down=k_down,  # SL 배수 (vol 기준)
    horizon_bars=h  # 수직 배리어
)
# 출력: 1 (Up), -1 (Down), 0 (Vertical)
# [V8.4] Long-Only: -1 → 0 변환
```

**학습**
```python
MLGuard.train(features=X_train, targets=y_train)
# - LightGBM Binary Classification (1 vs 0)
# - 70/30 Walk-Forward Split + Purge
# - is_unbalance=True
```

**예측**
```python
MLGuard.predict(X_test, threshold=0.5, max_prob=0.9)
# - raw_prob: 모델 확률
# - signal: threshold 통과 시 1, 아니면 0
# - scale: 확신도 기반 포지션 크기
```

### 🔴 문제점

| 문제 | 설명 | 영향도 |
|------|------|--------|
| **라벨 품질 의존성** | k, h 파라미터가 라벨 품질 결정 → RL과 분리됨 | 심각 |
| **In-Sample 학습** | 70% 학습 → 과적합 가능성 | 중간 |
| **단순 Binary** | Buy vs Hold만 학습 → 진입 타이밍 세부 정보 손실 | 중간 |
| **피처-라벨 불일치** | RL이 선택한 피처와 라벨링 기준 무관 | 높음 |

---

## 3️⃣ L1: 백테스트 및 평가

### 현재 구현

**Backtest**
```python
run_signal_backtest(price_df, results_df, risk_budget, cost_bps)
# - 실제 가격으로 손익 계산
# - trade_count, win_rate, total_return, mdd 추출
```

**Validation (Hard Gates)**
```python
validate_sample(metrics):
    1. trade_count >= VAL_MIN_TRADES (30)
    2. exposure_ratio >= VAL_MIN_EXPOSURE (2%)
    3. mdd_pct <= VAL_MAX_MDD_PCT (50%)
    4. VAL_WINRATE_MIN <= win_rate <= VAL_WINRATE_MAX
    5. reward_risk >= 0.5  # NEW
```

**Scoring**
```python
score_sample(metrics):
    1. CAGR × SCORE_W_CAGR (200)
    2. R:R 보너스 (25~50)
    3. MDD × SCORE_W_MDD (30)
    4. 거래 부족 점진적 패널티
    5. 승률 80%+ 패널티 (500)
```

### 🔴 문제점

| 문제 | 설명 | 영향도 |
|------|------|--------|
| **보상 지연** | 전체 백테스트 후 보상 → RL 학습 신호 희미 | 높음 |
| **단일 보상** | 복잡한 전략을 단일 숫자로 평가 → 정보 손실 | 중간 |
| **Credit Assignment** | 어떤 지표/파라미터가 성과에 기여했는지 불명확 | 심각 |

---

## 4️⃣ 발전 방향 (Evolution)

### 현재 구현

```
1. ε-greedy 탐색 (ε=0.3 → 0.05)
2. Q-Value 업데이트: Q(s,a) ← Q(s,a) + α(r + γ·max(Q') - Q(s,a))
3. D3QN: Experience Replay + Target Network
4. Curriculum: Stage 1→4 점진적 목표 상향
```

### 🔴 문제점

| 문제 | 설명 | 영향도 |
|------|------|--------|
| **State-Action 괴리** | State는 시장 상태, Action은 전략 유형 → 직접 연결 약함 | 높음 |
| **느린 수렴** | 6개 Action × 7개 State = 42 Q-Value만 학습 | 중간 |
| **변화 감지 불가** | 시장 변화 시 기존 Q-Value 무효화 | 높음 |

---

## 🔧 핵심 개선 제안

### 1. **Action Space 재설계** ⭐⭐⭐

**현재**: 고수준 전략 유형 (6개)
**문제**: 지표 선택이 랜덤이므로 RL 학습 무의미

**제안 A: Multi-Head Action (계층적)**
```python
Action = (IndicatorType, ParameterBucket, RiskProfile)
# 예: (RSI, [14-21], AGGRESSIVE)
# 조합 수: 20 × 5 × 4 = 400
```

**제안 B: Continuous Action (PPO/SAC 전환)**
```python
Action = [weight_trend, weight_momentum, weight_vol, rsi_window_norm, ...]
# 연속값 → 실제 파라미터로 매핑
```

### 2. **피처 선택 RL 통합** ⭐⭐⭐

**현재**: RL → 전략 유형 → 랜덤 피처
**문제**: RL이 구체적 피처를 제어 못함

**제안: Direct Feature Selection**
```python
# State: 시장 상태 + 최근 성과
# Action: 피처 ID 또는 피처 조합
# Reward: 해당 피처 사용 전략의 성과

class FeatureSelector(D3QNAgent):
    actions = registry.list_all()  # 모든 피처 ID
    
    def select_features(self, regime, n=3):
        # 상위 N개 Q-value 피처 선택
        q_values = self.get_q_values(regime)
        top_n = np.argsort(q_values)[-n:]
        return [self.actions[i] for i in top_n]
```

### 3. **Reward Decomposition** ⭐⭐

**현재**: 단일 총합 보상
**문제**: 어떤 요소가 성과에 기여했는지 불명확

**제안: Intrinsic Reward 추가**
```python
def compute_decomposed_reward(metrics):
    # Extrinsic: 최종 성과
    r_ext = total_return / 50  # [-1, 1]
    
    # Intrinsic: 학습 유도 신호
    r_trade_activity = min(n_trades / target, 1.0)  # 거래 활동
    r_rr_quality = (reward_risk - 1) / 1  # 손익비 개선
    r_regime_align = regime_match_trades / n_trades  # 레짐 일치
    
    # 최종: Extrinsic 점차 증가
    alpha = min(1.0, epoch / 1000)  # Curriculum Annealing
    return (1-alpha) * (r_trade_activity + r_rr_quality + r_regime_align) + alpha * r_ext
```

### 4. **라벨링-RL 통합** ⭐⭐⭐

**현재**: 라벨 파라미터 (k, h) 별도 관리
**문제**: RL이 라벨링 기준을 최적화 못함

**제안: Meta-Labeling RL**
```python
# RL이 라벨링 파라미터도 결정
action = {
    "strategy_type": "TREND_FOLLOWING",
    "k_up": 2.0,   # RL이 학습
    "k_down": 1.0,  # RL이 학습
    "horizon": 15,  # RL이 학습
}
# 이렇게 하면 전략과 라벨이 일치
```

### 5. **Opponent Modeling (시장 적응)** ⭐

**현재**: 시장 변화 시 기존 학습 무효화
**문제**: 과거 Q-Value가 현재 무효할 수 있음

**제안: Contextual Bandit + Forgetting**
```python
# Q-Value에 시간 감쇠 적용
decay = 0.995 ** days_since_update
effective_q = stored_q * decay + prior * (1 - decay)

# 또는 Regime별 별도 Q-Table
q_tables = {
    "BULL_RUN": QTable(),
    "PANIC": QTable(),
    ...
}
```

---

## 📊 구체적 코드 개선안

### 1. MetaAgent 개선: Direct Feature Control

```python
# src/l3_meta/agent.py 개선안

class MetaAgent:
    def __init__(self, ...):
        # 기존 strategy_rl 외에 feature_rl 추가
        all_features = registry.list_all()
        self.feature_rl = D3QNAgent(
            storage_path, 
            actions=[f.feature_id for f in all_features],
            model_name="d3qn_feature.pt"
        )
    
    def _construct_genome_from_action(self, action_name, regime):
        # 기존: 랜덤 피처 선택
        # 개선: RL로 피처 선택
        
        n_features = random.randint(
            config.GENOME_FEATURE_COUNT_MIN,
            config.GENOME_FEATURE_COUNT_MAX
        )
        
        selected_ids = []
        for _ in range(n_features):
            feature_id, _ = self.feature_rl.get_action(regime)
            if feature_id not in selected_ids:
                selected_ids.append(feature_id)
        
        # 파라미터 진화는 유지
        genome = {}
        for fid in selected_ids:
            meta = self.registry.get(fid)
            params = self._evolve_params(meta, action_name, regime)
            genome[fid] = params
        
        return genome
    
    def learn(self, reward, next_regime, policy_spec, metrics):
        # 기존 strategy_rl 학습
        self.strategy_rl.update(reward, ...)
        
        # 추가: feature_rl 학습 (피처별 기여도 기반)
        if metrics and policy_spec.feature_genome:
            feat_importance = metrics.get("feature_importance", {})
            for fid in policy_spec.feature_genome.keys():
                # 기여도 높은 피처에 더 높은 보상
                contrib = feat_importance.get(fid, 0)
                feat_reward = reward * (0.5 + contrib * 0.5)
                self.feature_rl.update_single(fid, feat_reward, next_regime)
```

### 2. Parameter Evolution 개선: Bayesian Optimization

```python
# src/l3_meta/param_optimizer.py (신규)

from skopt import gp_minimize
from skopt.space import Integer, Real

class ParamOptimizer:
    """
    Bayesian Optimization으로 파라미터 탐색.
    랜덤 대신 이전 결과 기반 탐색.
    """
    def __init__(self, storage_path):
        self.history = {}  # (feature_id, param_name) -> [(value, reward)]
    
    def suggest_params(self, feature_meta, prev_reward=None):
        if prev_reward is not None:
            self._update_history(feature_meta.feature_id, prev_params, prev_reward)
        
        space = []
        for p in feature_meta.params:
            if p.param_type == "int":
                space.append(Integer(p.min, p.max, name=p.name))
            elif p.param_type == "float":
                space.append(Real(p.min, p.max, name=p.name))
        
        if self._has_enough_history(feature_meta.feature_id):
            # Bayesian Optimization
            result = gp_minimize(
                lambda x: -self._expected_reward(x),
                space,
                n_calls=1,
                x0=self._get_init_points(feature_meta.feature_id)
            )
            return dict(zip([p.name for p in feature_meta.params], result.x))
        else:
            # 초기 랜덤 탐색
            return self._random_sample(feature_meta)
```

### 3. Reward Shaper 개선: Hindsight Experience Replay

```python
# src/l3_meta/reward_shaper.py 확장

class HindsightRewardShaper(RewardShaper):
    """
    거래 실패에서도 학습할 수 있도록 Hindsight Reward 생성.
    """
    def compute_hindsight_rewards(self, metrics, trades):
        """
        각 거래의 결과를 분석하여 개별 보상 생성.
        """
        hindsight_rewards = []
        
        for trade in trades:
            # 개별 거래 분석
            if trade["return_pct"] > 0:
                # 성공: 진입 조건 강화
                r = 0.1 + trade["return_pct"] / 100
            else:
                # 실패: 진입 조건을 반대로 했으면?
                # "이 상황에서는 진입 안 했어야" → Hold에 보상
                r = -0.1 + abs(trade["return_pct"]) / 200  # 덜 나쁨
            
            hindsight_rewards.append({
                "timestamp": trade["entry_time"],
                "reward": r,
                "action_taken": 1,  # Buy
                "hindsight_action": 0 if trade["return_pct"] < 0 else 1,
            })
        
        return hindsight_rewards
```

---

## 🎯 우선순위 Action Items

| 순위 | 작업 | 예상 효과 | 난이도 |
|------|------|----------|--------|
| 1 | **Feature RL 추가** | 피처 선택 학습 가능 | 중 |
| 2 | **라벨 파라미터 RL 통합** | 전략-라벨 일관성 | 중 |
| 3 | **Reward Decomposition** | 학습 신호 명확화 | 하 |
| 4 | **Parameter Bayesian Opt** | 탐색 효율 향상 | 중 |
| 5 | **Continuous Action (PPO)** | 표현력 증가 | 상 |

---

## 💡 결론

현재 시스템의 **가장 큰 문제점**은:

> **RL이 전략 "유형"만 선택하고, 실제 피처/파라미터는 랜덤이다.**

이로 인해:
1. RL이 실제로 학습하는 정보가 거의 없음
2. 좋은 피처 조합 발견이 순수 운에 의존
3. 보상이 RL 결정과 직결되지 않음

**해결책**:
1. **RL을 피처 수준까지 확장** (Feature RL)
2. **라벨링 파라미터도 RL 제어**
3. **Reward Decomposition으로 학습 신호 명확화**
