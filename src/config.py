"""
Configuration Module - 계층형 Config 구조

이 모듈은 모든 설정값을 중앙에서 관리합니다.
환경에 따라 다른 설정을 적용할 수 있습니다.

[V9 Major Update] Single Source of Truth for Backtest
- Entry Threshold 대폭 완화 (0.55 → 0.50)
- 최소 거래 수 현실적 조정 (150 → 30)
- 모든 평가 지표는 Backtest 기반
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from src.shared.stage_schema import StageSpec

load_dotenv()

# [V20 Blackwell Support] Enable CUDA DSA for sm_120 architecture
os.environ["TORCH_USE_CUDA_DSA"] = "1"


@dataclass
class BaseConfig:
    """기본 설정 클래스"""
    
    # ----------------------------------------
    # Paths
    # ----------------------------------------
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    
    @property
    def LEDGER_DIR(self) -> Path:
        return self.BASE_DIR / "ledger"
    
    @property
    def ARTIFACT_DIR(self) -> Path:
        return self.LEDGER_DIR / "artifacts"
    
    @property
    def LOG_DIR(self) -> Path:
        return self.BASE_DIR / "logs"
    
    @property
    def FEATURE_REGISTRY_PATH(self) -> Path:
        return self.BASE_DIR / "data" / "features.json"

    # ----------------------------------------
    # Data Settings
    # ----------------------------------------
    TARGET_TICKER: str = field(default_factory=lambda: os.getenv("TARGET_TICKER", "QQQ"))
    DATA_START_DATE: str = field(default_factory=lambda: os.getenv("DATA_START_DATE", "2020-01-01"))
    DATA_SOURCE: str = field(default_factory=lambda: os.getenv("DATA_SOURCE", "yfinance"))
    
    MACRO_TICKERS: Dict[str, str] = field(default_factory=lambda: {
        "VIX": "^VIX",
        "VVIX": "^VVIX",
        "US10Y": "^TNX",
        "DX": "DX-Y.NYB",
        "HYG": "HYG",
        "USO": "USO",
        "BTC": "BTC-USD",
        "SOXX": "SOXX",
        "EEM": "EEM",
        "XLK": "XLK",
        "XLE": "XLE"
    })

    # ----------------------------------------
    # SA-RAG (Ontology Zone)
    # ----------------------------------------
    SA_CHUNK_SIZE_WORDS: int = field(default_factory=lambda: int(os.getenv("SA_CHUNK_SIZE_WORDS", "500")))
    SA_CHUNK_OVERLAP_WORDS: int = field(default_factory=lambda: int(os.getenv("SA_CHUNK_OVERLAP_WORDS", "200")))
    SA_TOP_K: int = field(default_factory=lambda: int(os.getenv("SA_TOP_K", "5")))
    SA_N_HOP: int = field(default_factory=lambda: int(os.getenv("SA_N_HOP", "2")))
    SA_NORMALIZATION_C: float = field(default_factory=lambda: float(os.getenv("SA_NORMALIZATION_C", "0.4")))
    SA_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("SA_THRESHOLD", "0.5")))
    SA_MAX_ENTITIES_PER_CHUNK: int = field(default_factory=lambda: int(os.getenv("SA_MAX_ENTITIES_PER_CHUNK", "6")))
    SA_MAX_CONTEXT_CHUNKS: int = field(default_factory=lambda: int(os.getenv("SA_MAX_CONTEXT_CHUNKS", "6")))

    # [V12.3] Ontology Calibration
    ONTOLOGY_CALIB_THRESHOLD: float = 0.08  # Threshold to start trusting ontology
    ONTOLOGY_CALIB_HYSTERESIS: float = 0.03 # Hysteresis for stability
    ONTOLOGY_CALIB_TEMP: float = 0.02       # Temperature for sigmoid blending (smaller = sharper)
    ONTOLOGY_MIX_MIN: float = 0.1          # Min mixing ratio (base search)
    ONTOLOGY_MIX_MAX: float = 0.6          # Max mixing ratio (when fully calibrated)
    
    # ----------------------------------------
    # Execution Settings
    # ----------------------------------------
    STRICT_MODE: bool = False
    MAX_EXPERIMENTS: int = field(default_factory=lambda: int(os.getenv("MAX_EXPERIMENTS", "0")))
    SLEEP_INTERVAL: int = field(default_factory=lambda: int(os.getenv("SLEEP_INTERVAL", "1")))
    USE_FAST_MODE: bool = True
    
    PARALLEL_ENABLED: bool = True
    PARALLEL_MAX_WORKERS: int = field(
        default_factory=lambda: int(os.getenv("PARALLEL_MAX_WORKERS", str(os.cpu_count())))
    )
    PARALLEL_BATCH_SIZE: int = field(
        default_factory=lambda: int(os.getenv("PARALLEL_BATCH_SIZE", "0"))
    )
    PARALLEL_CHUNK_SIZE: int = field(
        default_factory=lambda: int(os.getenv("PARALLEL_CHUNK_SIZE", "10"))
    )
    PARALLEL_BACKEND: str = "loky"
    PARALLEL_TIMEOUT: int = 600
    
    # [V18] Device SSOT Policy
    # mode: "auto" | "cpu" | "cuda" | "mps"
    DEVICE_MODE: str = field(default_factory=lambda: os.getenv("DEVICE_MODE", "auto"))
    DEVICE_CPU_ONLY: bool = field(default_factory=lambda: os.getenv("DEVICE_CPU_ONLY", "False").lower() == "true")
    DEVICE_FORCE_FALLBACK: bool = False # Set to True if init fails during runtime
    
    RL_BATCH_LEARNING: bool = True
    
    # ----------------------------------------
    # RL Hyperparameters
    # ----------------------------------------
    RL_ALPHA: float = field(default_factory=lambda: float(os.getenv("RL_ALPHA", "0.1")))
    RL_GAMMA: float = field(default_factory=lambda: float(os.getenv("RL_GAMMA", "0.9")))
    # [V10] Epsilon 스케줄 개선 - 느린 decay로 탐색 기간 연장
    # [V11.2] Epsilon 스케줄 조정 - 초기 탐색 0.7, 느린 감쇠 0.998
    RL_EPSILON_START: float = field(default_factory=lambda: float(os.getenv("RL_EPSILON_START", "0.7")))
    RL_EPSILON_DECAY: float = field(default_factory=lambda: float(os.getenv("RL_EPSILON_DECAY", "0.998")))
    RL_EPSILON_MIN: float = field(default_factory=lambda: float(os.getenv("RL_EPSILON_MIN", "0.05")))
    RL_EPSILON_MAX: float = 0.8
    
    # [V10] Epsilon Reheat - 정책 고착 방지
    RL_EPSILON_REHEAT_ENABLED: bool = True
    RL_EPSILON_REHEAT_PERIOD: int = 200     # 정체 감지 및 리히트 체크 주기
    RL_EPSILON_REHEAT_VALUE: float = 0.7    # 리히트 시 복구할 Epsilon 값
    
    # ----------------------------------------
    # [V11.4] D3QN Settings (Deep RL)
    # ----------------------------------------
    D3QN_ENABLED: bool = field(default_factory=lambda: os.getenv("D3QN_ENABLED", "True").lower() == "true")
    D3QN_HIDDEN_DIM: int = 256
    D3QN_LEARNING_RATE: float = 1e-4
    D3QN_GAMMA: float = 0.95
    D3QN_TAU: float = 0.005        # Soft Update parameter
    D3QN_BUFFER_SIZE: int = 10000
    D3QN_BATCH_SIZE: int = 64
    D3QN_MIN_BUFFER_SIZE: int = 500
    D3QN_UPDATE_FREQ: int = 4
    D3QN_TARGET_UPDATE_FREQ: int = 100
    D3QN_EPSILON: float = 0.5      # Initial epsilon (if not using RL_EPSILON)
    D3QN_EPSILON_DECAY: float = 0.995
    D3QN_EPSILON_MIN: float = 0.05
    D3QN_REHEAT_PERIOD: int = 200
    D3QN_REHEAT_EPSILON: float = 0.3
    STATE_WINDOW_SIZE: int = 20
    STATE_FEATURE_DIM: int = 12
    
    # [V11.2] Gate Rejection Settings
    RL_REJECT_SCORE: float = -50.0          # Gate 실패 시 부여할 강력한 패널티 점수
    RL_SKIP_LEARNING_ON_REJECTION: bool = False  # True면 실패한 경험은 학습 데이터에서 제외 (비권장)

    # Reward Variance Watchdog
    REWARD_STD_WINDOW: int = field(default_factory=lambda: int(os.getenv("REWARD_STD_WINDOW", "200")))
    REWARD_STD_MIN: float = field(default_factory=lambda: float(os.getenv("REWARD_STD_MIN", "0.001")))
    
    # =========================================================
    # [V11] Reward Shaping - 설계도 기반 전면 개선
    # =========================================================
    # [주력 성분]
    REWARD_W_RETURN: float = 5.0         # [V11] 수익률 가중치 상향 (Alpha 강조)
    REWARD_W_RR: float = 1.0             
    REWARD_W_TOP_TRADES: float = 1.0     # [V11] 상위 트레이드 기여도 가중치 상향 (Quality 강조)
    REWARD_W_REGIME_TRADE: float = 0.5   # [V11] 레짐 필터 가중치 (Activity Gate로 활용됨)
    
    # [보조 성분]
    REWARD_W_TRADES: float = 1.5         # [V11] 적극성 유도
    REWARD_W_MDD: float = 0.5            
    REWARD_W_SHARPE: float = 0.1         
    REWARD_W_STABILITY: float = 0.1      
    
    # [스케일 및 임계값]
    # [V11.2] Alpha 벤치마크 (Buy & Hold CAGR ~15-20% 대비)
    REWARD_BENCHMARK_RETURN: float = 15.0  # 연간 벤치마크 수익률 (%)
    REWARD_RETURN_SCALE: float = 100.0     
    
    REWARD_SHARPE_SCALE: float = 2.0
    REWARD_MDD_THRESHOLD: float = 25.0   
    REWARD_MIN_TRADES: int = 10          # Stage 1: 10
    REWARD_TARGET_TRADES: int = 200      
    REWARD_MAX_TRADES: int = 1000        # 선형 구간 상한. 이후 비선형 패널티.
    
    # [V11] 손익비(R:R) 목표
    REWARD_TARGET_RR: float = 1.5        # 목표 R:R (이상이면 보너스)
    REWARD_EXCELLENT_RR: float = 2.0     # 우수 R:R (최대 보너스)
    
    # [V11] 상위 트레이드 기여도 목표
    REWARD_TOP_TRADE_PCT: float = 0.2          # 상위 20%
    REWARD_TOP_TRADE_CONTRIB_TARGET: float = 0.6  # 60% 이상 기여하면 보너스

    # ----------------------------------------
    # Rejection Scoring (Decomposed Penalty Vector)
    # ----------------------------------------
    REJECT_BASE_PENALTY_BY_STAGE: dict = field(default_factory=lambda: {
        1: -15.0,
        2: -30.0,
        3: -50.0,
    })
    REJECT_REASON_PENALTIES: dict = field(default_factory=lambda: {
        "FAIL_MIN_TRADES": -10.0,
        "FAIL_LOW_EXPOSURE": -8.0,
        "FAIL_LOW_RETURN": -12.0,
        "FAIL_WINRATE_LOW": -8.0,
        "FAIL_WINRATE_HIGH": -10.0,
        "FAIL_MDD_BREACH": -20.0,
        "FAIL_PF": -10.0,
        "FAIL_LUCKY_STRIKE": -15.0,
        "FAIL_SIGNAL_DEGENERATE": -8.0,
    })
    REJECT_DISTANCE_PENALTY_WEIGHTS: dict = field(default_factory=lambda: {
        "FAIL_MIN_TRADES": -12.0,
        "FAIL_LOW_EXPOSURE": -10.0,
        "FAIL_LOW_RETURN": -12.0,
        "FAIL_WINRATE_LOW": -8.0,
        "FAIL_WINRATE_HIGH": -8.0,
        "FAIL_MDD_BREACH": -20.0,
        "FAIL_PF": -8.0,
        "FAIL_LUCKY_STRIKE": -12.0,
        "FAIL_SIGNAL_DEGENERATE": -10.0,
    })
    REJECT_SCORE_FLOOR: float = -30.0  # [V12.3] Softened floor (-100 -> -30) to prevent death spirals
    REJECT_SOFT_PENALTY_SCALE: float = 0.5
    REJECT_NEAR_PASS_MAX_FAILURES: int = 2
    REJECT_NEAR_PASS_MAX_DISTANCE: float = 0.35

    # ----------------------------------------
    # Signal Degeneracy Thresholds
    # ----------------------------------------
    SIGNAL_DEGENERATE_MIN_TRADES: int = 5
    SIGNAL_DEGENERATE_MIN_ENTRY_RATE: float = 0.002
    SIGNAL_DEGENERATE_MIN_PCT_IN_MARKET: float = 0.01

    # ----------------------------------------
    # Genome Evolution Settings
    # ----------------------------------------
    GENOME_FEATURE_COUNT_MIN: int = 1
    GENOME_FEATURE_COUNT_MAX: int = 5
    
    ENTRY_THRESHOLD_MIN: float = 0.45
    ENTRY_THRESHOLD_MAX: float = 0.55
    
    PARAM_DEFENSIVE_WINDOW_RATIO: float = 0.5
    PARAM_SCALPING_MAX_WINDOW: int = 20
    PARAM_PANIC_MAX_WINDOW: int = 30
    
    # [V11] Stage별 지표 풀 - 점진적 확장
    # "인기 지표 우대" ❌ / "초반 학습 가속용 Prior" ⭕
    STAGE_FEATURE_POOLS: dict = field(default_factory=lambda: {
        1: ["RSI", "SMA", "EMA", "BB", "ATR", "VOLUME_SMA"],  # 기본 지표
        2: ["RSI", "SMA", "EMA", "BB", "ATR", "VOLUME_SMA", 
            "STOCH", "ROC", "ADX", "CCI", "MFI"],  # 확장 지표
        3: ["*"],  # Stage 3+: 전체 개방
        4: ["*"],
    })
    
    # [V11] 카테고리 슬롯 제한 - 조건 희소성 폭발 방지
    # 각 카테고리에서 최대 N개 지표만 선택
    CATEGORY_SLOTS: dict = field(default_factory=lambda: {
        "TREND": 1,      # Trend 1개
        "MOMENTUM": 1,   # Momentum 1개
        "VOLATILITY": 1, # Volatility 1개
        "VOLUME": 1,     # Volume/Flow 1개
    })
    
    # [V11] 시장 컨텍스트 필수 피처 - 모든 전략에 강제 포함
    # RL이 선택하는 피처 + 이 피처들 = 최종 피처셋
    # 목적: 시장 상태 무시 전략 방지, 레짐 인식 학습 유도
    MANDATORY_CONTEXT_FEATURES: bool = True  # 필수 포함 활성화
    
    CONTEXT_FEATURES: dict = field(default_factory=lambda: {
        # 변동성 컨텍스트 - "지금 시장이 공포인가?"
        "CTX_VIX": {
            "description": "VIX 기반 변동성 상태",
            "compute": "vix_proxy",  # 실제 VIX 또는 실현변동성
        },
        # 추세 컨텍스트 - "지금 상승장인가 하락장인가?"
        "CTX_TREND": {
            "description": "가격 vs MA 추세 상태",
            "compute": "trend_score",
        },
        # 모멘텀 컨텍스트 - "최근 힘이 어디로 가는가?"
        "CTX_MOMENTUM": {
            "description": "단기 수익률 모멘텀",
            "compute": "momentum_roc",
        },
        # 레짐 컨텍스트 - "시장 국면 복합 점수"
        "CTX_REGIME": {
            "description": "복합 시장 레짐 점수",
            "compute": "regime_score",
        },
    })
    
    # ----------------------------------------
    # Risk Parameter Ranges
    # ----------------------------------------
    RISK_K_UP_MIN: float = 0.5
    RISK_K_UP_MAX: float = 4.0
    RISK_K_DOWN_MIN: float = 0.3
    RISK_K_DOWN_MAX: float = 2.5
    RISK_HORIZON_MIN: int = 3
    RISK_HORIZON_MAX: int = 30
    RISK_EST_DAILY_VOL: float = 0.015
    
    MAX_LEVERAGE_MIN: float = 0.5
    MAX_LEVERAGE_MAX: float = 1.5
    STOP_LOSS_MIN: float = 0.005
    STOP_LOSS_MAX: float = 0.05

    # ----------------------------------------
    # Evaluation Settings (Backtest 기반)
    # ----------------------------------------
    EVAL_RISK_JITTER_PCT: float = 0.05
    EVAL_RISK_SAMPLES_FAST: int = 3
    EVAL_RISK_SAMPLES_REDUCED: int = 10
    EVAL_RISK_SAMPLES_FULL: int = 30

    EVAL_WINDOW_COUNT_FAST: int = 2
    EVAL_WINDOW_COUNT_REDUCED: int = 3
    EVAL_WINDOW_COUNT_FULL: int = 6
    EVAL_MIN_WINDOW_BARS: int = 120
    EVAL_LOWER_QUANTILE: float = 0.2

    EVAL_FAST_LOOKBACK_BARS: int = 600
    EVAL_REDUCED_SLICE_BARS: int = 500
    EVAL_REDUCED_SLICES: int = 3

    EVAL_FAST_TOP_PCT: float = 0.5
    EVAL_REDUCED_TOP_PCT: float = 0.3
    EVAL_FULL_TOP_K: int = 0
    EVAL_FULL_EVERY: int = 3

    # ----------------------------------------
    # Evaluation Cache Settings
    # ----------------------------------------
    SIGNAL_CACHE_MAXSIZE: int = field(default_factory=lambda: int(os.getenv("SIGNAL_CACHE_MAXSIZE", "256")))
    BACKTEST_CACHE_MAXSIZE: int = field(default_factory=lambda: int(os.getenv("BACKTEST_CACHE_MAXSIZE", "128")))
    BACKTEST_CACHE_TTL_SEC: int = field(default_factory=lambda: int(os.getenv("BACKTEST_CACHE_TTL_SEC", "3600")))

    # ----------------------------------------
    # TradingView Calibration (V12-BT Validation)
    # ----------------------------------------
    TV_CALIBRATION_DATA_DIR: str = field(default_factory=lambda: os.getenv("TV_CALIBRATION_DATA_DIR", "data/calibration"))
    TV_CALIBRATION_REFERENCE_DIR: str = field(default_factory=lambda: os.getenv("TV_CALIBRATION_REFERENCE_DIR", "data/calibration"))
    TV_CALIBRATION_EXECUTION: str = field(default_factory=lambda: os.getenv("TV_CALIBRATION_EXECUTION", "next_open"))
    TV_CALIBRATION_TP_SL_PRIORITY: str = field(default_factory=lambda: os.getenv("TV_CALIBRATION_TP_SL_PRIORITY", "stop_first"))
    TV_CALIBRATION_ALLOW_SAME_BAR_REENTRY: bool = field(default_factory=lambda: os.getenv("TV_CALIBRATION_ALLOW_SAME_BAR_REENTRY", "False").lower() == "true")
    TV_CALIBRATION_PYRAMIDING: int = field(default_factory=lambda: int(os.getenv("TV_CALIBRATION_PYRAMIDING", "0")))
    TV_CALIBRATION_POSITION_MODE: str = field(default_factory=lambda: os.getenv("TV_CALIBRATION_POSITION_MODE", "equity_pct"))
    TV_CALIBRATION_POSITION_VALUE: float = field(default_factory=lambda: float(os.getenv("TV_CALIBRATION_POSITION_VALUE", "1.0")))
    TV_CALIBRATION_COMMISSION_TYPE: str = field(default_factory=lambda: os.getenv("TV_CALIBRATION_COMMISSION_TYPE", "percent"))
    TV_CALIBRATION_COMMISSION_VALUE: float = field(default_factory=lambda: float(os.getenv("TV_CALIBRATION_COMMISSION_VALUE", "0.0")))
    TV_CALIBRATION_SLIPPAGE_MODE: str = field(default_factory=lambda: os.getenv("TV_CALIBRATION_SLIPPAGE_MODE", "ticks"))
    TV_CALIBRATION_SLIPPAGE_VALUE: float = field(default_factory=lambda: float(os.getenv("TV_CALIBRATION_SLIPPAGE_VALUE", "0.0")))
    TV_CALIBRATION_TICK_SIZE: float = field(default_factory=lambda: float(os.getenv("TV_CALIBRATION_TICK_SIZE", "0.01")))
    TV_CALIBRATION_TP_PCT: float = field(default_factory=lambda: float(os.getenv("TV_CALIBRATION_TP_PCT", "0.002")))
    TV_CALIBRATION_SL_PCT: float = field(default_factory=lambda: float(os.getenv("TV_CALIBRATION_SL_PCT", "0.002")))
    TV_CALIBRATION_MAX_HOLD_BARS: int = field(default_factory=lambda: int(os.getenv("TV_CALIBRATION_MAX_HOLD_BARS", "30")))
    TV_CALIBRATION_TOL_PRICE_PCT: float = field(default_factory=lambda: float(os.getenv("TV_CALIBRATION_TOL_PRICE_PCT", "0.0005")))
    TV_CALIBRATION_TOL_FINAL_EQUITY_PCT: float = field(default_factory=lambda: float(os.getenv("TV_CALIBRATION_TOL_FINAL_EQUITY_PCT", "0.005")))
    TV_CALIBRATION_TOL_MDD_PCT: float = field(default_factory=lambda: float(os.getenv("TV_CALIBRATION_TOL_MDD_PCT", "0.001")))
    TV_CALIBRATION_CAUSE_MAP: Dict[str, List[str]] = field(default_factory=lambda: {
        "TRADE_COUNT_MISMATCH": [
            "signal_shift_mismatch",
            "crossover_logic_mismatch",
            "pyramiding_or_reentry_mismatch",
        ],
        "ENTRY_BAR_MISMATCH": [
            "orders_at_close_mismatch",
            "signal_shift_mismatch",
            "entry_price_source_mismatch",
        ],
        "EXIT_BAR_MISMATCH": [
            "orders_at_close_mismatch",
            "signal_shift_mismatch",
            "intrabar_exit_mismatch",
        ],
        "ENTRY_PRICE_MISMATCH": [
            "execution_timing_mismatch",
            "slippage_application_mismatch",
            "price_source_mismatch",
        ],
        "EXIT_REASON_MISMATCH": [
            "tp_sl_priority_mismatch",
            "intrabar_path_mismatch",
            "time_stop_definition_mismatch",
        ],
        "EXIT_PRICE_MISMATCH": [
            "tp_sl_priority_mismatch",
            "slippage_application_mismatch",
            "orders_at_close_mismatch",
        ],
        "FEE_MISMATCH": [
            "commission_unit_mismatch",
            "commission_side_application_mismatch",
            "position_sizing_mismatch",
        ],
        "SLIPPAGE_MISMATCH": [
            "slippage_unit_mismatch",
            "slippage_side_application_mismatch",
            "tick_size_mismatch",
        ],
        "FINAL_EQUITY_MISMATCH": [
            "position_sizing_mismatch",
            "equity_update_timing_mismatch",
            "fee_slippage_application_mismatch",
        ],
        "MDD_MISMATCH": [
            "equity_curve_timing_mismatch",
            "mark_to_market_mismatch",
            "partial_fill_assumption_mismatch",
        ],
    })

    # =========================================================
    # [V10] Validation Layer (Hard Gates) - 절대 생존 기준 강화
    # =========================================================
    # [Gate 1] 거래 활동성 - 커리큘럼 연동 (기본값은 Stage 1)
    VAL_MIN_TRADES: int = 15          # [V11.2] Raw 거래 문턱 (최소 활동)
    VAL_MIN_VALID_TRADES: int = 10    # [V11.2] Regime-Aligned 유효 거래 문턱
    VAL_MIN_VALID_RATIO: float = 0.5  # [V11.2] 전체 거래 중 유효 거래 최소 비율
    
    VAL_MIN_TRADES_PER_YEAR: int = 6 # 연간 최소 거래 (30 -> 6)
    VAL_MIN_EXPOSURE: float = 0.02    # [Gate 2] 최소 시장 참여율 낮춤 (5% -> 2%)
    VAL_MAX_MDD_PCT: float = 50.0     # [Gate 3] 생존 최대 손실폭

    # [V10] 승률 범위 제한 (추세 추종 전략을 위해 하한 대폭 완화)
    VAL_WINRATE_MIN: float = 0.30     # [V10] 최소 승률 하향 (0.35 -> 0.30)
    VAL_WINRATE_MAX: float = 0.85     # 최대 승률
    VAL_WINRATE_PENALTY_THRESHOLD: float = 0.80  # 이 이상은 패널티 시작

    # =========================================================
    # [vNext] Evaluation Layer (Scoring) - 순위 결정 로직
    # =========================================================
    # [1순위] 복리 성과 (압도적)
    SCORE_W_CAGR: float = 200.0       # [V10] 수익률 영향력 2배 강화 (100 -> 200)
    
    # [2순위] 리스크 패널티
    SCORE_W_MDD: float = 30.0         # MDD 1%당 30점 감점
    SCORE_W_VOL: float = 5.0          # 변동성 패널티
    
    # [3순위] 거래 활동 보정 - 부족 시 '감점' (보상이 아닌 페널티)
    SCORE_TARGET_TRADES: int = 100    # 목표 거래 수
    SCORE_PENALTY_LOW_TRADE: float = 0.2  # [V10] 감점 대폭 완화 (50 -> 0.2)
    
    # [4순위] 승률 과적합 강력 견제
    SCORE_TARGET_WINRATE_MIN: float = 0.45
    SCORE_TARGET_WINRATE_MAX: float = 0.70
    SCORE_PENALTY_HIGH_WR: float = 500.0   # [V10] 승률 80% 초과 시 강력 패널티
    SCORE_WINRATE_PENALTY_START: float = 0.80  # 패널티 시작 승률
    
    # =========================================================
    # [V16] AutoTuner Settings
    # =========================================================
    AUTOTUNE_ENABLED: bool = True
    AUTOTUNE_HISTORY_SIZE: int = 20         # 분석할 최근 배치 수
    AUTOTUNE_RIGID_THRESHOLD: float = 0.05  # 5% 미만 개선 시 정체로 판단
    AUTOTUNE_PASS_RATE_TARGET: float = 0.15 # 목표 통과율 (Stage 2)
    AUTOTUNE_DIVERSITY_TARGET: float = 0.4  # 목표 Jaccard 거리 (평균)
    AUTOTUNE_INTERVENTION_M_BATCHES: int = 10 # 개입 효과 검증 기간
    
    # Mutation Levers
    AUTOTUNE_MUTATION_BASE: float = 0.1
    AUTOTUNE_INDICATOR_SWAP_PROB: float = 0.2

    # =========================================================
    # [V11] Curriculum Learning - 설계도 기반 단계별 목표
    # "500%는 목표이지 출발선이 아니다"
    # =========================================================
    # [V13-PRO] Complexity Management
    SCORE_W_COMPLEXITY: float = 2.0    # Token complexity penalty
    
    CURRICULUM_ENABLED: bool = True
    CURRICULUM_STAGES: Dict[int, StageSpec] = field(default_factory=lambda: {
        1: StageSpec(
            stage_id=1,
            name="Discovery",
            target_return_pct=10.0,     # [V12.3] Lowered target (15 -> 10) to focus on structural integrity
            alpha_floor=-10.0,          # [V12.3] Very soft alpha floor
            min_trades_per_year=4.0,    # [V12.3] Lowered (6 -> 4) to allow emerging signals
            max_mdd_pct=50.0,           # [V12.3] Relaxed MDD (40 -> 50)
            min_profit_factor=0.9,      # [V12.3] Relaxed PF (1.0 -> 0.9)
            and_terms_range=(1, 2),
            quantile_bias="center",
            wf_splits=3,
            wf_gate_mode="soft",
            exploration_slot=0.5,       # [V12.3] High exploration
            reject_base_penalty=-10.0,  # [V12.3] Low penalty to encourage trying
            signal_degeneracy_mode="soft"
        ),
        2: StageSpec(
            stage_id=2,
            name="Survival",
            target_return_pct=30.0,
            alpha_floor=0.0,
            min_trades_per_year=12.0,
            max_mdd_pct=25.0,
            min_profit_factor=1.1,
            and_terms_range=(2, 3),
            quantile_bias="spread",
            wf_splits=3,
            wf_gate_mode="soft",
            exploration_slot=0.2,
            reject_base_penalty=-30.0,
            signal_degeneracy_mode="soft"
        ),
        3: StageSpec(
            stage_id=3,
            name="Deployment",
            target_return_pct=60.0,
            alpha_floor=15.0,           # High edge required
            min_trades_per_year=15.0,
            max_mdd_pct=15.0,
            min_profit_factor=1.3,
            and_terms_range=(2, 4),
            quantile_bias="tail",
            wf_splits=5,
            wf_gate_mode="hard",
            exploration_slot=0.1,
            reject_base_penalty=-50.0,
            signal_degeneracy_mode="hard"
        )
    })
    CURRICULUM_CURRENT_STAGE: int = 1
    CURRICULUM_STAGE_UP_THRESHOLD: int = 5
    
    # [V14] Failure Taxonomy Tree
    # Categories: SIGNAL, EDGE, RISK, COMPLEXITY, DATA
    FAILURE_TAXONOMY: dict = field(default_factory=lambda: {
        "SIGNAL_ISSUE": ["FAIL_MIN_TRADES", "FAIL_LOW_EXPOSURE", "FAIL_ZERO_EXPOSURE", "FAIL_OVER_EXPOSURE", "FAIL_SIGNAL_DEGENERATE"],
        "EDGE_ISSUE": ["FAIL_LOW_RETURN", "FAIL_NEG_ALPHA", "FAIL_PF", "FAIL_WINRATE_LOW", "FAIL_WINRATE_HIGH"],
        "RISK_ISSUE": ["FAIL_MDD_BREACH", "FAIL_LUCKY_STRIKE", "FAIL_WORST_WINDOW"],
        "COMPLEXITY_ISSUE": ["FAIL_COMPLEXITY_HIGH", "FAIL_AST_DEPTH"],
        "DATA_ISSUE": ["FAIL_QUANTILE_COLLISION", "FAIL_MISSING_DATA", "FAIL_FEATURE_COLLAPSE"]
    })

    # [V14] Stage Health Diagnostic Rules
    # (min_pass_rate, target_rejection_rate, target_annual_trades)
    STAGE_HEALTH_RULES: dict = field(default_factory=lambda: {
        1: { # Discovery
            "rejection_rate_range": (0.3, 0.8),
            "median_tpy_range": (10, 50),
            "min_pass_rate": 0.2
        },
        2: { # Survival
            "rejection_rate_range": (0.6, 0.95),
            "median_tpy_range": (8, 30),
            "min_pass_rate": 0.05,
            "min_median_excess_pa": -1.0
        },
        3: { # Deployment
            "rejection_rate_range": (0.8, 0.99),
            "min_pass_rate": 0.01,
            "min_oos_pass_rate": 0.01
        }
    })

    # [V14] Stagnation Detection
    STAGNATION_BATCH_WINDOW: int = 5
    STAGNATION_MIN_PROGRESS: float = 0.01 # Reward improvement floor
    
    # =========================================================
    # [V18] LogicTree Strict Mode - Feature Matching Control
    # =========================================================
    # 학습 모드에서는 침묵 실패(silent failure)를 금지하고,
    # 미매칭/모호성 발생 시 명시적으로 REJECT 처리
    LOGICTREE_STRICT: bool = True  # True = 학습 모드 (엄격), False = 운영 모드 (관대)
    LOGICTREE_FUZZY_MATCH: bool = True  # prefix 기반 fuzzy matching 허용 여부
    
    # 모호성 처리 정책: "error" | "warn_pick_first" | "warn_pick_value"
    # - error: 학습 모드에서 즉시 reject
    # - warn_pick_first: 경고 후 첫 번째 컬럼 선택 (운영 모드용)
    # - warn_pick_value: 경고 후 __value 또는 알파벳 순 선택
    LOGICTREE_AMBIGUOUS_POLICY: str = "error"
    
    # =========================================================
    # [V11.3] New Features: Stability & Diversity
    # =========================================================
    
    # ----------------------------------------
    # 1. Warm-start Pool (Baseline)

    # [V14] Stage Health Diagnostic Rules
    # (min_pass_rate, target_rejection_rate, target_annual_trades)
    STAGE_HEALTH_RULES: dict = field(default_factory=lambda: {
        1: { # Discovery
            "rejection_rate_range": (0.3, 0.8),
            "median_tpy_range": (10, 50),
            "min_pass_rate": 0.2,
            "min_median_excess_pa": -10.0 # [V12.3]
        },
        2: { # Survival
            "rejection_rate_range": (0.6, 0.95),
            "median_tpy_range": (8, 30),
            "min_pass_rate": 0.05,
            "min_median_excess_pa": -1.0
        },
        3: { # Deployment
            "rejection_rate_range": (0.8, 0.99),
            "min_pass_rate": 0.01,
            "min_oos_pass_rate": 0.01,
            "min_median_excess_pa": 5.0 # [V12.3]
        }
    })

    # [V14] Stagnation Detection
    STAGNATION_BATCH_WINDOW: int = 5
    STAGNATION_MIN_PROGRESS: float = 0.01 # Reward improvement floor
    
    # =========================================================
    # [V18] LogicTree Strict Mode - Feature Matching Control
    # =========================================================
    # 학습 모드에서는 침묵 실패(silent failure)를 금지하고,
    # 미매칭/모호성 발생 시 명시적으로 REJECT 처리
    LOGICTREE_STRICT: bool = True  # True = 학습 모드 (엄격), False = 운영 모드 (관대)
    LOGICTREE_FUZZY_MATCH: bool = False # [V18] SSOT 우선: 추측 금지
    
    # 모호성 처리 정책: "error" | "warn_pick_first" | "warn_pick_value"
    LOGICTREE_AMBIGUOUS_POLICY: str = "error"
    LOGICTREE_FAIL_ACTION: str = "INVALID_SPEC" # REJECTED보다 높은 단계의 시스템 오류로 처리
    
    # =========================================================
    # [V11.3] New Features: Stability & Diversity
    # =========================================================
    
    # ----------------------------------------
    # 1. Warm-start Pool (Baseline)
    # ----------------------------------------
    WARM_START_N1: int = 200         # Gen 1~N1: baseline 50%
    WARM_START_N2: int = 500         # Gen N1~N2: baseline 30%
    WARM_START_BASE_DIR: Path = field(default_factory=lambda: Path("data/baselines"))
    
    # ----------------------------------------
    # 2. Diversity Controller
    # ----------------------------------------
    DIVERSITY_K: int = 30            # 최종 생존 개수
    DIVERSITY_JACCARD_TH: float = 0.7 # 이 이상이면 너무 유사
    DIVERSITY_PARAM_DIST_TH: float = 0.1
    
    # ----------------------------------------
    # 3. Anti-Luck Filter
    # ----------------------------------------
    ANTILUCK_TOP1_SHARE_MAX: float = 0.60
    ANTILUCK_TOP3_SHARE_MAX: float = 0.85
    ANTILUCK_MODE: str = "soft"  # [V12.3] "soft" | "hard". Stage-dependent override.

    # ----------------------------------------
    # Replay Buffer Tagging
    # ----------------------------------------
    REPLAY_TAGGED_ENABLED: bool = True
    REPLAY_TAG_SAMPLE_RATIOS: dict = field(default_factory=lambda: {
        "PASS": 0.5,
        "NEAR_PASS": 0.3,
        "HARD_FAIL": 0.2,
    })
    
    # ----------------------------------------
    # 4. Walk-forward Consistency Gate
    # ----------------------------------------
    WF_GATE_ENABLED: bool = True
    WF_SPLITS_STAGE1: int = 3
    WF_SPLITS_STAGE2: int = 3
    WF_SPLITS_STAGE3: int = 5
    WF_ALPHA_FLOOR_STAGE1: float = -0.02
    WF_ALPHA_FLOOR_STAGE2: float = -0.01
    WF_ALPHA_FLOOR_STAGE3: float = 0.00

    # (Deprecated but kept for backward compatibility)
    EVAL_ENTRY_THRESHOLD: float = 0.50
    EVAL_ENTRY_MAX_PROB: float = 0.9
    EVAL_SCORE_MIN: float = -15.0     # [V10] 하한값 현실화 (-9999 -> -15)
    
    # [Backward Compatibility Aliases] Old names -> New names
    EVAL_MIN_TRADES: int = 30         # -> VAL_MIN_TRADES
    EVAL_MAX_DD_PCT: float = 50.0     # -> VAL_MAX_MDD_PCT
    EVAL_TRADE_TARGET: int = 100      # -> SCORE_TARGET_TRADES
    EVAL_SCORE_W_RETURN: float = 100.0
    EVAL_SCORE_W_RETURN_Q: float = 0.8
    EVAL_SCORE_W_WINRATE: float = 5.0
    EVAL_SCORE_W_MDD: float = 30.0
    EVAL_SCORE_W_VOL: float = 5.0
    EVAL_SCORE_W_RR: float = 0.1
    EVAL_SCORE_W_TRADE: float = 2.0
    EVAL_SCORE_W_VIOLATION: float = 100.0
    EVAL_RISK_SCALE_FLOOR: float = 0.25
    EVAL_TARGET_WINRATE_MIN: float = 0.45
    EVAL_TARGET_WINRATE_MAX: float = 0.70

    # ----------------------------------------
    # Transaction Cost
    # ----------------------------------------
    DEFAULT_COST_BPS: int = 10
    
    # ----------------------------------------
    # Data Window
    # ----------------------------------------
    DEFAULT_LOOKBACK_BARS: int = 500
    
    # ----------------------------------------
    # [vAlpha+] EAGL Constants (Economic Alpha Guidance Layer)
    # ----------------------------------------
    EAGL_ENABLED: bool = True
    AOS_TAU: float = 0.1             # Softmax temperature for Exploration Budget
    AOS_WEIGHT_RETURN_COST: float = 0.4
    AOS_WEIGHT_FREQUENCY: float = 0.3
    AOS_WEIGHT_CONSISTENCY: float = 0.3
    
    # Reward Components V2
    REWARD_REPEATABILITY_W: float = 0.2
    REWARD_COST_SURVIVAL_W: float = 0.3
    REWARD_STABILITY_W: float = 0.2
    
    # CRM (Conditional Revival)
    CRM_DORMANT_THRESHOLD: int = 5   # Failure count before dormancy
    CRM_REVIVAL_MIN_ALPHA: float = 0.02
    
    def __post_init__(self):
        """디렉토리 생성"""
        self.LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        self.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DevelopmentConfig(BaseConfig):
    """개발 환경용 설정"""
    
    EVAL_RISK_SAMPLES_FAST: int = 2
    EVAL_RISK_SAMPLES_REDUCED: int = 5
    EVAL_RISK_SAMPLES_FULL: int = 15
    
    EVAL_FAST_LOOKBACK_BARS: int = 300
    EVAL_MIN_WINDOW_BARS: int = 60
    
    RL_EPSILON_START: float = field(default_factory=lambda: float(os.getenv("RL_EPSILON_START", "0.3")))
    RL_EPSILON_DECAY: float = field(default_factory=lambda: float(os.getenv("RL_EPSILON_DECAY", "0.99")))
    
    DEBUG_MODE: bool = True
    VERBOSE_LOGGING: bool = True


@dataclass
class ProductionConfig(BaseConfig):
    """프로덕션 환경용 설정"""
    
    STRICT_MODE: bool = True
    
    EVAL_RISK_SAMPLES_FAST: int = 5
    EVAL_RISK_SAMPLES_REDUCED: int = 15
    EVAL_RISK_SAMPLES_FULL: int = 50
    
    RL_EPSILON_START: float = field(default_factory=lambda: float(os.getenv("RL_EPSILON_START", "0.15")))
    RL_EPSILON_DECAY: float = field(default_factory=lambda: float(os.getenv("RL_EPSILON_DECAY", "0.998")))
    
    DEBUG_MODE: bool = False
    VERBOSE_LOGGING: bool = False


@dataclass
class TestConfig(BaseConfig):
    """테스트 환경용 설정"""
    
    EVAL_RISK_SAMPLES_FAST: int = 1
    EVAL_RISK_SAMPLES_REDUCED: int = 2
    EVAL_RISK_SAMPLES_FULL: int = 3
    
    EVAL_FAST_LOOKBACK_BARS: int = 100
    EVAL_MIN_WINDOW_BARS: int = 30
    
    RL_EPSILON_START: float = 0.5
    RL_EPSILON_DECAY: float = 0.9
    
    @property
    def LEDGER_DIR(self) -> Path:
        return self.BASE_DIR / "test_ledger"


_config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "test": TestConfig,
    "default": BaseConfig,
}

_active_config: Optional[BaseConfig] = None


def get_config(env: Optional[str] = None) -> BaseConfig:
    """환경에 맞는 Config 인스턴스를 반환합니다."""
    global _active_config
    
    if env is None:
        env = os.getenv("ENV", "default")
    
    config_class = _config_map.get(env.lower(), BaseConfig)
    return config_class()


def set_active_config(cfg: BaseConfig) -> None:
    """활성 Config를 직접 설정합니다 (테스트용)."""
    global _active_config
    _active_config = cfg


# Default Singleton Instance
config = get_config()
