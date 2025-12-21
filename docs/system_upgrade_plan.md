# Vibe System Upgrade Plan: Regime-Adaptive Architecture

## 1. Regime-Specific Multi-Genome (장세 인식형 멀티 게놈)
**목표**: '평균의 함정' 탈피. 시장 국면별 최적화된 별도 모델 운용.
- [ ] **Data Layer**: FRED/YFinance 연동하여 Macro Data (VIX, 금리, 달러) 수집 파이프라인 구축.
- [ ] **Meta Layer**: `RegimeDetector`를 4대 국면(Goldilocks, Stagflation, Panic, Bull Run) 판별기로 고도화.
- [ ] **Execution Layer**: 백테스팅 엔진이 날짜별 Regime에 따라 동적으로 Strategy(Genome)를 스위칭하도록 개편.
- [ ] **Training**: 국면별로 데이터를 슬라이싱하여 별도 학습 진행.

## 2. Probabilistic ML Entry (확률 기반 진입)
**목표**: 단순 Threshold(RSI<30)가 아닌, 승률 확률(Confidence Score) 기반 진입.
- [ ] **Feature Engineering**: 21개 지표 + 매크로 데이터를 ML 모델 인풋 벡터로 변환.
- [ ] **Model**: LightGBM/XGBoost 기반의 진입 확률 예측 모델 구축.
- [ ] **Signal**: `Confidence > 0.7` 일 때만 진입하는 필터링 로직 구현.

## 3. Macro & Volatility Features (데이터 차원 확장)
**목표**: 시장의 '환경'을 모델에 주입.
- [ ] **Integration**: `DataLoader`에 VIX, Treasury Yield, Put/Call Ratio 등 연동.
- [ ] **Preprocessing**: 서로 다른 타임라인 데이터의 정렬 및 결측치 처리 (Forward Fill).
- [ ] **Input**: Meta-Agent와 Tactical-Model에 해당 피처 주입.
