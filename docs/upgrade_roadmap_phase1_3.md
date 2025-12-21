# Vibe System Upgrade Roadmap: Macro-Driven Adaptive Intelligence

## Phase 1. Data Layer Expansion (The Eyes) - [Partially Done]
**목표**: 시장의 '외부 환경'을 인식할 수 있는 데이터 파이프라인 완성.
**현재 진행상황**: VIX, US10Y, DX(달러) 수집 및 RegimeDetector 1차 개편 완료.

### [추가 제안: 필수 매크로 데이터]
더 정교한 판단을 위해 다음 데이터 추가를 강력 추천합니다.

1.  **Yield Curve Spread (10Y - 2Y)**
    *   **티커**: `T10Y2Y` (FRED) 또는 `^TEN2` (계산 필요)
    *   **이유**: 경기 침체의 가장 강력한 선행 지표. 역전(음수)되면 `STAGFLATION` 또는 `RECESSION` 모드 트리거.
2.  **High Yield Bond Spread (HYG vs Agg)**
    *   **티커**: `HYG` (High Yield ETF), `LQD` (Investment Grade)
    *   **이유**: 회사채 스프레드가 벌어지면 '신용 위험' 신호 → 주식 비중 축소.
3.  **Oil Price (WTI)**
    *   **티커**: `CL=F` (Crude Oil Future)
    *   **이유**: 인플레이션의 주범. 오일 급등 + 주가 하락은 전형적인 `STAGFLATION`.
4.  **Sector Rotation (XLK vs XLE)**
    *   **티커**: `XLK` (Tech), `XLE` (Energy), `XLF` (Financial)
    *   **이유**: 기술주가 약세인데 에너지가 강하면 '순환매' 장세. 기술주 원툴 전략 방지.

---

## Phase 2. Context-Aware Meta Agent (The Brain)
**목표**: Regime별로 완전히 다른 '전략 유전자(Genome)'를 관리하는 멀티 페르소나 에이전트 구축.

### [개선 계획]
1.  **Multi-Q-Table Architecture**
    *   기존: 1개의 Q-Table이 모든 상황을 학습.
    *   **변경**: `Q_Table_BULL`, `Q_Table_BEAR`, `Q_Table_PANIC` 등 상태별 별도 Q-Table(또는 뇌) 분리.
    *   **이유**: "상승장의 눌림목 매수"와 "하락장의 눌림목 매수"는 정반대의 결과를 낳음. 섞이면 안 됨.
2.  **Dynamic Regime Memory**
    *   **구현**: `MetaAgent`가 현재 Regime을 감지하면, 해당 Regime 전용 `Action Space`를 활성화.
    *   *예시: PANIC 모드에서는 'Leverage' 사용 금지, 'Short' 액션 활성화.*

---

## Phase 3. Probabilistic Entry (The Execution)
**목표**: 0/1 (진입/대기) 이분법을 넘어 '확률(Probability)' 기반의 베팅 크기 조절.

### [개선 계획]
1.  **LightGBM 'Gatekeeper' Model**
    *   **역할**: 전략이 "매수" 신호를 보내도, ML 모델이 "성공 확률 40%"라고 판단하면 진입 차단.
    *   **구현**: `Strategy.execute()` 직전에 `Gatekeeper.predict(current_features)` 호출.
2.  **Context Features Injection**
    *   **피처**: 기술적 지표 21개 + **Phase 1에서 수집한 매크로 데이터**를 ML 모델에 인풋으로 주입.
    *   *효과: "RSI가 낮지만(매수 신호), VIX가 40이고 하이일드 스프레드가 치솟고 있으니(매크로 악재) 진입 금지" 판단 가능.*

---

## [Action Plan Summary]
1.  **Data**: 오일(CL=F), 하이일드(HYG), 섹터(XLK/XLE) 데이터 소스 추가 (`loader.py`).
2.  **Meta Agent**: Regime별 Q-Table 분리 및 저장/로드 로직 개편 (`agent.py`, `q_learner.py`).
3.  **Tactical**: LightGBM 모델 학습 파이프라인 구축 및 실시간 추론 연동 (`ml_guard.py` 신설).
