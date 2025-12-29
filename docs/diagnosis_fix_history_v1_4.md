# 진단/수정 타임라인 (v1.0~v1.4)

이 문서는 지금까지 확인된 **원인**과 **수정 내용**을 시간순으로 정리한 기록이다.
모든 원인은 SSOT 관점에서 재현 가능한 증거를 남기고, 변경은 관측 가능한 지표로 검증했다.

## 0. 전제
- 관측 근거: `logs/observability/batch_reports.jsonl`
- 목표: 거래 가능한 정책을 구조적으로 생성하고, 그 위에서만 학습/승격이 일어나게 하는 것

## 1. 구조적 결함 진단 및 복구 (초기)
### 원인
1) FeatureRegistry 사전 컴파일 실패  
   - 상수/기본 로직이 로딩되지 않아 `entry_signal_rate_mean=0`, `idle_ratio_mean=1.0` 패턴 발생
2) Stage cooldown 고정  
   - batch_id 리셋과 last_change_batch 불일치로 승격/강등이 영구 봉인
3) 계측 오버헤드 음수  
   - 계측 중복 기록으로 병목 판단이 왜곡

### 수정
- FeatureRegistry 안정화 및 실패 로그 분리
- StageController cooldown 로직 정상화
- 계측 중복/오버헤드 음수 영역 분리

### 결과
- `entry_signal_rate_mean`이 0에서 회복
- Stage 전이 로그가 실제로 움직이기 시작
- 계측 로그가 병목 분석에 재사용 가능해짐

## 2. 상태-행동 불일치로 인한 Invalid 폭증 (정책 구조 결함)
### 원인
- 정책은 “시장 조건”만 보고 신호 생성
- 실행 단계에서 상태(FLAT/LONG/SHORT)와 충돌 → Invalid 구조적으로 필연
- reward collapse 발생(동일 패널티 몰림)

### 수정
- 정책 생성 단계에 상태 조건 포함  
  - ENTRY 조건: `STATE == FLAT`  
  - EXIT 조건: `STATE == LONG`
- invalid와 ignored 분리, invalid은 규약 위반으로만 집계

### 결과
- invalid_action_rate가 0에 수렴
- reward collapse 원인 분리 가능

## 3. Reward collapse 및 REJECT 패널티 분산 (v1.2)
### 원인
- 실패 정책의 reward가 동일값에 몰림
- REJECT가 단일 고정값 처리 → 분산 소멸

### 수정
- REJECT 패널티를 거리 기반 분산으로 변경
- 배치 리포트에 `fixed_penalty_top_value`, `reject_reason_distribution`, `reject_distance_stats` 추가
- reward collapse trend 계산 추가

### 결과
- `reward_unique_count` 증가
- collapse 빈도 하락

## 4. Stage2 목표 불일치 (min_flips 하드 강제, v1.3)
### 원인
- Stage2에서 FAIL_MIN_FLIPS가 하드 실패로 동작
- 추세/단방향 전략이 구조적으로 탈락 → 실패 원인 단일화

### 수정
- Stage2에서 FAIL_MIN_FLIPS를 soft 처리
- Stage2에서 signal_degeneracy를 hard로 전환
- Stage2 soft gate에서 FAIL_MIN_FLIPS 가중치 0

### 결과
- 실패 원인 단일화 완화
- Stage2에서 valid/gate_pass 발생 시작

## 5. SSOT 정합성 결함: hit-rate bounds vs min_entries (v1.4 핵심)
### 원인
- `min_entries_per_year` 요구치가 `ENTRY_HIT_RATE_BOUNDS.max`로는 물리적으로 달성 불가
- 결과: FAIL_MIN_TRADES가 구조적으로 지배

### 수정
1) SSOT 정합성 계산 도입  
   - required_entry_rate 계산  
   - feasibility_ok / margin 기록  
   - 배치 리포트에 `stage_constraints` 추가
2) 정합성 충돌 시 failure_mode가 REFACTOR로 전환되도록 연결
3) 생성기 편향을 required_entry_rate 기준으로 보정
4) Stage별 hit-rate max 상향으로 정합성 확보

### 결과
- `feasibility_ok`가 10/10 유지
- `decision=continue` 고정
- Stage2 → Stage3 승격 발생

## 6. 동적 최소 hit-rate 필터 적용 (v1.4 추가)
### 원인
- FAIL_MIN_TRADES 비중이 여전히 높음
- prefilter가 고정 min_rate만 사용 → 거래 밀도 개선이 느림

### 수정
- 동적 최소 hit-rate 적용  
  - `min_rate_effective = max(min_rate, required_entry_rate * ENTRY_RATE_MIN_SCALE)`
- `recommended_min_rate`, `min_rate_scale` 리포트 추가

### 결과
- trade_median 평균 상승(약 23)
- valid_count 평균 3.0, gate_pass_count 평균 3.0
- reward_unique_count 평균 4.8 유지
- Stage 승격 안정화(2→3)

## 7. 현재 상태 요약 (최근 10배치 기준)
- feasibility_ok: 10/10
- trade_median 평균: 23.7
- entry_signal_rate_mean 평균: 0.0637
- valid_count 평균: 3.0
- reward_collapse: 1/10
- 주요 실패 분포: FAIL_LOW_RETURN / FAIL_MIN_TRADES / FAIL_MIN_FLIPS

현재는 “거래 발생” 단계는 복구 완료 상태이며,
병목이 수익/리스크 계열(LOW_RETURN, MDD, PF)로 이동한 상태다.
