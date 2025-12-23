# 전략 학습 시스템 V11 개선 설계 - 구현 완료

## 🎯 최종 목표
- 거래를 "회피하지 않는" 전략
- 유리한 국면에서만 진입
- 승률은 낮아도 손익비로 크게 이김
- 장기 복리 기준 **5년 500% 이상**
- 과최적화·도망 전략 자동 제거

---

## 📁 수정된 파일 목록

### 1. 신규 생성
- `src/l3_meta/curriculum_controller.py` - 단계별 학습 관리자

### 2. 전면 개선
- `src/l3_meta/reward_shaper.py` - 다면적 보상 함수
- `src/config.py` - 설정값 대폭 업데이트
- `src/l1_judge/evaluator.py` - Validation Layer 강화
- `src/orchestration/infinite_loop.py` - Curriculum 통합

---

## 🧠 아키텍처 구현 현황

### 1️⃣ Validation Layer (Hard Gate) ✅
**위치**: `src/l1_judge/evaluator.py` - `validate_sample()`

| Gate | 설명 | 기준 |
|------|------|------|
| Trade Count | 최소 거래 수 | VAL_MIN_TRADES (30) |
| Exposure | 시장 참여율 | VAL_MIN_EXPOSURE (2%) |
| MDD | 생존 리스크 | VAL_MAX_MDD_PCT (50%) |
| Win Rate High | 과적합 의심 | VAL_WINRATE_MAX (85%) |
| Win Rate Low | 구조 결함 | VAL_WINRATE_MIN (30%) |
| **R:R (NEW)** | 손익비 검증 | R:R >= 0.5 |

### 2️⃣ Reward Shaper (학습 유도) ✅
**위치**: `src/l3_meta/reward_shaper.py`

| 보상 유형 | 가중치 | 설명 |
|-----------|--------|------|
| **Return** | 2.0 | 수익률 - Profit is King |
| **R:R** | 1.0 | 손익비 보상 (목표 1.5+, 우수 2.0+) |
| Top Trades | 0.5 | 상위 20% 기여도 60%+ 보너스 |
| Regime Trade | 0.3 | 레짐 일치 거래 보상 |
| Trades | 0.6 | 거래 활동성 보상 |
| MDD Penalty | 0.3 | MDD 20%+ 패널티 |
| **Winrate Penalty** | - | 80%+ 승률 패널티 |

**핵심**: 무거래 = -2.0 (최악의 점수)

### 3️⃣ Evaluator (순위 결정) ✅
**위치**: `src/l1_judge/evaluator.py` - `score_sample()`

| 순위 | 항목 | 설명 |
|------|------|------|
| 1 | **CAGR** | 복리 성과 (압도적 영향) |
| 2 | **R:R 보너스** | 2.0+ = +50점, 1.5+ = +25점 |
| 3 | MDD/Vol | 리스크 패널티 |
| 4 | 거래 부족 | 목표 미달 시 감점 |
| 5 | 승률 과도 | 80%+ 시 강력 패널티 |

### 4️⃣ Curriculum Controller (단계 관리) ✅
**위치**: `src/l3_meta/curriculum_controller.py`

| Stage | 수익률 | 거래 수 | MDD | R:R | 설명 |
|-------|--------|---------|-----|-----|------|
| 1 | 50%+ | 50+ | 50% | 1.0 | 기본 활동성 + 생존 확보 |
| 2 | 150%+ | 100+ | 45% | 1.2 | 수익성 개선 + 손익비 의식 |
| 3 | 250%+ | 150+ | 40% | 1.5 | 안정적 고수익 + R:R 최적화 |
| 4 | **500%+** | 200+ | 35% | 2.0 | 5년 장기 복리 전략 |

**자동 승급**: 10개 통과 시 다음 Stage

### 5️⃣ Evolution Engine (연동) ✅
- 상위 점수 전략 유지
- Validation 탈락 전략 절대 진입 불가
- D3QN/Q-Table 학습에 R:R 메트릭 전달

### 6️⃣ 지표(Feature) 전략 ✅
**위치**: `src/config.py` - `STAGE_FEATURE_POOLS`, `CATEGORY_SLOTS`

**Stage별 지표 풀**:
- Stage 1: RSI, SMA, EMA, BB, ATR, VOLUME_SMA
- Stage 2: + STOCH, ROC, ADX, CCI, MFI
- Stage 3+: 전체 개방

**카테고리 슬롯** (조건 희소성 방지):
- TREND: 1개
- MOMENTUM: 1개
- VOLATILITY: 1개
- VOLUME: 1개

---

## ⚙️ 새로운 Config 설정값

```python
# Reward Shaping (V11)
REWARD_W_RETURN = 2.0        # 수익률
REWARD_W_RR = 1.0            # 손익비
REWARD_W_TOP_TRADES = 0.5    # 상위 트레이드
REWARD_W_REGIME_TRADE = 0.3  # 레짐 일치
REWARD_TARGET_RR = 1.5       # R:R 목표
REWARD_EXCELLENT_RR = 2.0    # R:R 우수

# Curriculum Learning (V11)
CURRICULUM_ENABLED = True
CURRICULUM_STAGE_UP_THRESHOLD = 10

# Feature Selection (V11)
STAGE_FEATURE_POOLS = {...}
CATEGORY_SLOTS = {...}
```

---

## ❌ 절대 금지 사항 (구현됨)

1. **승률 중심 회귀** → 승률 보너스 폐지, 과도 승률 패널티
2. **초반 거래 수 억제** → 초반 Stage는 거래 수 기준 낮음
3. **손실 패널티 재강화** → MDD 패널티 완화 (0.5 → 0.3)

---

## 🚀 학습 진행 단계 (절대 순서)

1. ✅ **공격성 확보** (완료 - 초기 설정)
2. ✅ **방향성 필터** (V11 - 레짐 일치 거래 보상)
3. ✅ **손익비 / 한 방 구조** (V11 - R:R 보상, 상위 트레이드 기여도)
4. ⏳ **거래 수 억제** (Stage 3+ 자동 적용)
5. ⏳ **복리·장기 안정성** (Stage 4 목표)

---

## 📊 최종 한 문장 요약

> 이 설계는 "평가를 통과하는 전략"이 아니라  
> **"시장에서 돈을 버는 전략만 살아남도록"** 시스템을 강제한다.
