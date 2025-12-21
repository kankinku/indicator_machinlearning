# Template-based Meta-Optimization Trading System (Vibe Coding Edition)

> **"금융 머신러닝의 3대 함정(Overfitting, Look-ahead Bias, High Cost)을 원천 차단하는 자율주행 트레이딩 연구소"**

이 프로젝트는 무작위 지표 조합이 아닌, **검증된 전략 템플릿(Strategy Template)** 위에서 메타 에이전트(L3)가 **시장 상황(Regime)**에 맞춰 최적의 파라미터를 스스로 학습하고 진화시키는 시스템입니다.

---

## 🏗️ Architecture Overview

시스템은 3개의 지능 계층(Intelligence Layer)과 1개의 실행 엔진으로 구성됩니다.

### 1. L3 Meta-Agent (The Strategist)
*   **Regime Detector:** 현재 시장이 추세장인지, 횡보장인지, 과열 구간인지 판단합니다. (ADX, Vol Cone, Bollinger Breach 활용)
*   **Contextual Bandit:** 현재 Regime에 가장 적합한 전략 템플릿(예: 하락장엔 T08 방어형)을 제안합니다.
*   **Evolutionary Strategy:** 과거의 성공한 실험 데이터를 바탕으로 파라미터를 미세 조정(Mutation)하여 최적값을 탐색합니다.

### 2. L2 Tactical Engine (The Builder)
*   **Smart Feature Factory:** **Look-ahead Bias**를 방지하면서도 당일 종가 데이터를 최대한 활용하는 Smart Lagging 기술이 적용되었습니다.
*   **Triple Barrier Labeling:** 고정된 수익률이 아닌, 시장 변동성(Volatility)에 비례하는 동적 목표가/손절가를 설정합니다.
*   **Cost-Aware Modeling:** 수수료와 슬리피지를 반영한 Net PnL을 기준으로 모델을 학습시킵니다.

### 3. L1 Judge (The Auditor)
*   **CPCV (Combinatorial Purged CV):** 시계열 데이터의 특성을 고려하여, **"학습하지 않은 미래 데이터"**에 대해서만 예측 성능을 평가합니다. (In-Sample Overfitting 원천 차단)
*   **Rigorous Verdict:** PBO(확률적 과최적화), Sharpe, Drawdown, Turnover 등을 종합적으로 평가하여 승인(Approved) 여부를 결정합니다.

---

## 🚀 Getting Started

### 1. Installation
필요한 라이브러리를 설치합니다.
```bash
pip install pandas numpy scikit-learn ta joblib
```

### 2. Quick Start (Infinite Loop)
아무런 준비 없이도 바로 자율 실험을 시작할 수 있습니다. (더미 데이터 자동 생성)
```bash
python src/orchestration/infinite_loop.py
```
*   시스템이 자동으로 데이터를 로드하고,
*   현재 시장 상황(Regime)을 분석한 뒤,
*   적절한 전략을 수립하여 실험(Experiment)을 수행하고,
*   결과를 `ledger/` 디렉토리에 저장합니다.

### 3. Integration Test
전체 파이프라인의 건전성을 확인하려면 다음 테스트를 실행하세요.
```bash
python tests/integration_test.py
```

---

## 📂 Project Structure

```
src/
├── features/           # Feature Engineering (Smart Lagging)
│   └── factory.py      # Feature Factory
├── l1_judge/           # Evaluation Logic
│   ├── cpcv.py         # Cross-Validation Engine
│   └── risk_engine.py  # Drawdown & Risk Check
├── l2_sl/              # Supervised Learning Engine
│   ├── labeling/       # Triple Barrier Labeling
│   ├── direction/      # GBDT Model
│   └── artifacts.py    # Model & Result Saver
├── l3_meta/            # Meta-Learning Agent
│   ├── agent.py        # Bandit & ES Logic
│   └── detectors/      # Regime Detection
├── orchestration/      # Execution Loop
│   ├── run_experiment.py # Single Experiment Pipeline
│   └── infinite_loop.py  # Main Autonomous Loop
├── ledger/             # Experiment Database (JSONL + Artifacts)
└── templates/          # Strategy Registry (T01~T08)
```

---

## 📊 Outputs (Ledger System)

모든 실험은 `ledger/` 폴더에 완벽하게 기록됩니다.

1.  **`experiments.jsonl`**: 모든 실험의 요약 기록 (파라미터, 성과, 판결).
2.  **`artifacts/{UUID}.json`**: 해당 실험의 상세 메타데이터.
3.  **`artifacts/{UUID}.model.joblib`**: 학습된 AI 모델 파일 (바이너리).
4.  **`artifacts/{UUID}_results.csv`**: 상세 백테스팅 타임로그 (Date, Predicted, Probability, Actual, Net PnL).

---

## 💡 Key Features for Quants

*   **Transaction Cost Model:** 편도 5bp(기본값) 수수료를 차감한 Net PnL로 평가하여, 잦은 매매로 인한 손실을 방지합니다.
*   **OOS Assembly:** Cross-Validation의 예측값만을 모아 Equity Curve를 그리므로, 실제 라이브 트레이딩과 유사한 성과를 보여줍니다.
*   **Explainable Verdict:** 전략이 왜 실패했는지(예: `CPCV_WORST_TOO_LOW`, `DD_LIMIT_BREACH`) 명확한 이유(Reason Code)를 제시합니다.
