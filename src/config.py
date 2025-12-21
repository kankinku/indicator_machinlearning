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
        "US10Y": "^TNX",  # CBOE 10 Year Treasury Note Yield
        "DX": "DX-Y.NYB",  # US Dollar Index
        "HYG": "HYG",      # High Yield Corporate Bond ETF (Credit Risk)
        "USO": "USO",      # United States Oil Fund (Energy/Inflation Proxy)
        "XLK": "XLK",      # Tech Sector (Growth)
        "XLE": "XLE"       # Energy Sector (Value/Inflation)
    }
    
    # Execution Settings
    MAX_EXPERIMENTS: int = int(os.getenv("MAX_EXPERIMENTS", "0")) # 0 means infinite
    SLEEP_INTERVAL: int = int(os.getenv("SLEEP_INTERVAL", "1"))
    
    # RL Hyperparameters (Configuration Separation)
    RL_ALPHA: float = float(os.getenv("RL_ALPHA", "0.1"))    # Learning Rate
    RL_GAMMA: float = float(os.getenv("RL_GAMMA", "0.9"))    # Discount Factor
    RL_EPSILON_START: float = float(os.getenv("RL_EPSILON_START", "0.2"))
    RL_EPSILON_DECAY: float = float(os.getenv("RL_EPSILON_DECAY", "0.995"))
    RL_EPSILON_MIN: float = float(os.getenv("RL_EPSILON_MIN", "0.05"))
    
    def __post_init__(self):
        # Ensure directories exist
        self.LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        self.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

# Singleton Instance
config = AppConfig()
