
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import config
from src.features.registry import FeatureRegistry
from src.contracts import FeatureMetadata, TunableParamSpec

def populate_initial_population():
    """
    Populates the registry with the 'Adam & Eve' features (Basic TA-Lib indicators).
    """
    registry = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH))
    registry.initialize()
    
    print(f"Populating initial features into {config.FEATURE_REGISTRY_PATH}...")
    
    # 1. RSI (Momentum)
    rsi_code = """
import pandas as pd
import ta

class RSIIndicator:
    def compute(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        target = ta.momentum.RSIIndicator(df["close"], window=window)
        return pd.DataFrame({f"RSI_{window}": target.rsi()}, index=df.index)
"""
    rsi = FeatureMetadata(
        feature_id="MOMENTUM_RSI_V1",
        name="Relative Strength Index",
        category="MOMENTUM",
        description="Standard RSI indicator.",
        code_snippet=rsi_code,
        handler_func="RSIIndicator",
        params=[
            TunableParamSpec(name="window", param_type="int", min=2, max=100, default=14)
        ],
        source="builtin"
    )
    registry.register(rsi, overwrite=True)
    
    # 2. MACD (Trend)
    macd_code = """
import pandas as pd
import ta

class MACDIndicator:
    def compute(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        macd = ta.trend.MACD(df["close"], window_slow=slow, window_fast=fast, window_sign=signal)
        return pd.DataFrame({
            f"MACD_diff_{fast}_{slow}": macd.macd_diff(),
            f"MACD_sig_{fast}_{slow}": macd.macd_signal()
        }, index=df.index)
"""
    macd = FeatureMetadata(
        feature_id="TREND_MACD_V1",
        name="Moving Average Convergence Divergence",
        category="TREND",
        description="Standard MACD.",
        code_snippet=macd_code,
        handler_func="MACDIndicator",
        params=[
            TunableParamSpec(name="fast", param_type="int", min=2, max=50, default=12),
            TunableParamSpec(name="slow", param_type="int", min=10, max=200, default=26),
            TunableParamSpec(name="signal", param_type="int", min=2, max=50, default=9)
        ],
        source="builtin"
    )
    registry.register(macd, overwrite=True)
    
    # 3. Bollinger Bands (Volatility)
    bb_code = """
import pandas as pd
import ta

class BBIndicator:
    def compute(self, df: pd.DataFrame, window: int = 20, dev: float = 2.0) -> pd.DataFrame:
        bb = ta.volatility.BollingerBands(df["close"], window=window, window_dev=dev)
        return pd.DataFrame({
            f"BB_w_{window}": bb.bollinger_wband(),
            f"BB_p_{window}": bb.bollinger_pband()
        }, index=df.index)
"""
    bb = FeatureMetadata(
        feature_id="VOLATILITY_BB_V1",
        name="Bollinger Bands",
        category="VOLATILITY",
        description="Standard Bollinger Bands Width and %B.",
        code_snippet=bb_code,
        handler_func="BBIndicator",
        params=[
            TunableParamSpec(name="window", param_type="int", min=5, max=100, default=20),
            TunableParamSpec(name="dev", param_type="float", min=1.0, max=4.0, default=2.0)
        ],
        source="builtin"
    )
    registry.register(bb, overwrite=True)

    print("Success! Initial population created.")

if __name__ == "__main__":
    populate_initial_population()
