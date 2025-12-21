import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass(frozen=True)
class AppConfig:
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    LEDGER_DIR: Path = BASE_DIR / "ledger"
    ARTIFACT_DIR: Path = LEDGER_DIR / "artifacts"
    LOG_DIR: Path = BASE_DIR / "logs"
    FEATURE_REGISTRY_PATH: Path = BASE_DIR / "data" / "features.json"


    # Data Settings
    TARGET_TICKER: str = os.getenv("TARGET_TICKER", "QQQ")
    DATA_START_DATE: str = os.getenv("DATA_START_DATE", "2015-01-01")
    
    # Macro Data Source (YFinance Tickers)
    MACRO_TICKERS = {
        "VIX": "^VIX",
        "VVIX": "^VVIX",   # Volatility of Volatility
        "US10Y": "^TNX",  # CBOE 10 Year Treasury Note Yield
        "DX": "DX-Y.NYB",  # US Dollar Index
        "HYG": "HYG",      # High Yield Corporate Bond ETF (Credit Risk)
        "USO": "USO",      # United States Oil Fund (Energy/Inflation Proxy)
        "BTC": "BTC-USD",  # Crypto (Risk-on/Liquidity Proxy)
        "SOXX": "SOXX",    # Semiconductors (Tech Cycle)
        "EEM": "EEM",      # Emerging Markets (Global Growth)
        "XLK": "XLK",      # Tech Sector (Growth)
        "XLE": "XLE"       # Energy Sector (Value/Inflation)
    }
    
    # Execution Settings
    STRICT_MODE: bool = False  # If True, stops on error
    MAX_EXPERIMENTS: int = int(os.getenv("MAX_EXPERIMENTS", "0")) # 0 means infinite
    SLEEP_INTERVAL: int = int(os.getenv("SLEEP_INTERVAL", "1"))
    USE_FAST_MODE: bool = True
    
    # RL Hyperparameters (Configuration Separation)
    RL_ALPHA: float = float(os.getenv("RL_ALPHA", "0.1"))    # Learning Rate
    RL_GAMMA: float = float(os.getenv("RL_GAMMA", "0.9"))    # Discount Factor
    RL_EPSILON_START: float = float(os.getenv("RL_EPSILON_START", "0.2"))
    RL_EPSILON_DECAY: float = float(os.getenv("RL_EPSILON_DECAY", "0.995"))
    RL_EPSILON_MIN: float = float(os.getenv("RL_EPSILON_MIN", "0.05"))
    
    # ========================================
    # Risk Parameter Evolution Ranges
    # ========================================
    # These define the search space for risk parameters.
    # Each experiment samples from these ranges, and good combinations
    # are naturally selected through the reward mechanism.
    
    # Target Profit (k_up): Volatility multiplier for profit taking barrier
    # Higher = Let profits run more, Lower = Quick take profit
    RISK_K_UP_MIN: float = 0.5    # Conservative: ~0.5x volatility target
    RISK_K_UP_MAX: float = 4.0    # Aggressive: 4x volatility target (big moves)
    
    # Stop Loss (k_down): Volatility multiplier for stop loss barrier  
    # Higher = Wider stops (more room), Lower = Tight stops (quick cut)
    RISK_K_DOWN_MIN: float = 0.3  # Very tight stop
    RISK_K_DOWN_MAX: float = 2.5  # Wide stop (give room for recovery)
    
    # Max Holding Period (horizon): Bars before time-based exit
    # Higher = Longer holds (trend), Lower = Quick trades (momentum)
    RISK_HORIZON_MIN: int = 5     # Very short-term (intraday feel)
    RISK_HORIZON_MAX: int = 60    # Multi-week holds
    RISK_EST_DAILY_VOL: float = 0.015  # Used for TP/SL pct conversions

    # ========================================
    # Evaluation & Ranking Settings
    # ========================================
    # Risk sampling around the policy's target/stop
    EVAL_RISK_JITTER_PCT: float = 0.05  # +/- 5% (Respect model's choice)
    EVAL_RISK_SAMPLES_FAST: int = 3
    EVAL_RISK_SAMPLES_REDUCED: int = 10
    EVAL_RISK_SAMPLES_FULL: int = 30

    # Time-split evaluation (rolling windows)
    EVAL_WINDOW_COUNT_FAST: int = 2
    EVAL_WINDOW_COUNT_REDUCED: int = 3
    EVAL_WINDOW_COUNT_FULL: int = 6
    EVAL_MIN_WINDOW_BARS: int = 120
    EVAL_LOWER_QUANTILE: float = 0.2

    # Data reduction
    EVAL_FAST_LOOKBACK_BARS: int = 600
    EVAL_REDUCED_SLICE_BARS: int = 500
    EVAL_REDUCED_SLICES: int = 3

    # Ranking and pruning
    EVAL_FAST_TOP_PCT: float = 0.5
    EVAL_REDUCED_TOP_PCT: float = 0.3
    EVAL_FULL_TOP_K: int = 0  # 0 = keep all after reduced stage
    EVAL_FULL_EVERY: int = 3  # Run full evaluation every N batches

    # Score normalization & weights
    EVAL_RISK_SCALE_FLOOR: float = 0.25  # % floor to avoid tiny divisor
    EVAL_TRADE_TARGET: int = 150
    EVAL_SCORE_W_RETURN: float = 3.0
    EVAL_SCORE_W_RETURN_Q: float = 0.6
    EVAL_SCORE_W_WINRATE: float = 0.5
    EVAL_SCORE_W_MDD: float = 0.1
    EVAL_SCORE_W_VOL: float = 0.1
    EVAL_SCORE_W_RR: float = 0.2
    EVAL_SCORE_W_TRADE: float = 5.0
    EVAL_SCORE_W_VIOLATION: float = 0.8
    EVAL_MAX_DD_PCT: float = 20.0
    EVAL_MIN_TRADES: int = 20
    EVAL_ENTRY_THRESHOLD: float = 0.55
    EVAL_ENTRY_MAX_PROB: float = 0.9
    EVAL_SCORE_MIN: float = -999.0
    
    def __post_init__(self):
        # Ensure directories exist
        self.LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        self.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

# Singleton Instance
config = AppConfig()
