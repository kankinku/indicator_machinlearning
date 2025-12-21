
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class FeatureParamSchema:
    name: str
    param_type: str  # "int", "float", "choice"
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Any = None

@dataclass
class FeatureDefinition:
    id: str
    name: str
    category: str # "momentum", "volatility", "pattern", "volume"
    description: str
    params: List[FeatureParamSchema]

# --- Indicator Universe Definition ---
# This acts as the "Gene Pool" for the Agent.

INDICATOR_UNIVERSE = {
    "RSI": FeatureDefinition(
        id="RSI",
        name="Relative Strength Index",
        category="momentum",
        description="Measures the speed and change of price movements.",
        params=[
            FeatureParamSchema("window", "int", min=5, max=50, default=14),
            FeatureParamSchema("threshold_low", "int", min=20, max=40, default=30),
            FeatureParamSchema("threshold_high", "int", min=60, max=80, default=70),
        ]
    ),
    "MACD": FeatureDefinition(
        id="MACD",
        name="Moving Average Convergence Divergence",
        category="momentum",
        description="Trend-following momentum indicator.",
        params=[
            FeatureParamSchema("fast", "int", min=5, max=20, default=12),
            FeatureParamSchema("slow", "int", min=21, max=50, default=26),
            FeatureParamSchema("signal", "int", min=5, max=15, default=9),
        ]
    ),
    "BB": FeatureDefinition(
        id="BB",
        name="Bollinger Brands",
        category="volatility",
        description="Volatility bands placed above and below a moving average.",
        params=[
            FeatureParamSchema("window", "int", min=10, max=50, default=20),
            FeatureParamSchema("std_dev", "float", min=1.5, max=3.0, step=0.1, default=2.0),
        ]
    ),
    "ATR": FeatureDefinition(
        id="ATR",
        name="Average True Range",
        category="volatility",
        description="Measure of volatility.",
        params=[
            FeatureParamSchema("window", "int", min=5, max=30, default=14),
        ]
    ),
    "STOCH": FeatureDefinition(
        id="STOCH",
        name="Stochastic Oscillator",
        category="momentum",
        description="Momentum indicator comparing a particular closing price to a range of its prices.",
        params=[
            FeatureParamSchema("window", "int", min=5, max=30, default=14),
            FeatureParamSchema("smooth_k", "int", min=1, max=10, default=3),
            FeatureParamSchema("smooth_d", "int", min=1, max=10, default=3),
        ]
    ),
    "OBV": FeatureDefinition(
        id="OBV",
        name="On-Balance Volume",
        category="volume",
        description="Momentum indicator that uses volume flow to predict changes in stock price.",
        params=[] # No params
    ),
    "MA_CROSS": FeatureDefinition(
        id="MA_CROSS",
        name="Moving Average Crossover",
        category="trend",
        description="Signal when fast MA crosses slow MA.",
        params=[
            FeatureParamSchema("fast", "int", min=5, max=50, default=20),
            FeatureParamSchema("slow", "int", min=50, max=200, default=60),
            FeatureParamSchema("ma_type", "choice", choices=["sma", "ema"], default="sma"),
        ]
    ),
    # --- Momentum ---
    "CCI": FeatureDefinition(
        id="CCI",
        name="Commodity Channel Index",
        category="momentum",
        description="Identifies cyclical turns in commodities.",
        params=[
            FeatureParamSchema("length", "int", min=10, max=50, default=14),
            FeatureParamSchema("c", "float", min=0.01, max=0.05, step=0.005, default=0.015),
        ]
    ),
    "ROC": FeatureDefinition(
        id="ROC",
        name="Rate of Change",
        category="momentum",
        description="Pure momentum oscillator that measures the percentage change in price.",
        params=[
            FeatureParamSchema("length", "int", min=1, max=30, default=10),
        ]
    ),
    "WILLR": FeatureDefinition(
        id="WILLR",
        name="Williams %R",
        category="momentum",
        description="Momentum indicator moving between 0 and -100.",
        params=[
            FeatureParamSchema("length", "int", min=5, max=30, default=14),
        ]
    ),
    "MOM": FeatureDefinition(
        id="MOM",
        name="Momentum",
        category="momentum",
        description="Simple change in price.",
        params=[
            FeatureParamSchema("length", "int", min=1, max=30, default=10),
        ]
    ),
    "CMO": FeatureDefinition(
        id="CMO",
        name="Chande Momentum Oscillator",
        category="momentum",
        description="Calculates the difference between the sum of recent gains and losses.",
        params=[
            FeatureParamSchema("length", "int", min=5, max=30, default=14),
        ]
    ),
    
    # --- Trend ---
    "ADX": FeatureDefinition(
        id="ADX",
        name="Average Directional Index",
        category="trend",
        description="Measures the strength of a trend.",
        params=[
            FeatureParamSchema("length", "int", min=5, max=30, default=14),
            FeatureParamSchema("lensig", "int", min=5, max=30, default=14),
        ]
    ),
    "TRIX": FeatureDefinition(
        id="TRIX",
        name="Triple Exponential Average",
        category="trend",
        description="Smoothed momentum oscillator to identify trends.",
        params=[
            FeatureParamSchema("length", "int", min=10, max=50, default=30),
            FeatureParamSchema("signal", "int", min=5, max=20, default=9),
        ]
    ),
    "SUPER": FeatureDefinition(
        id="SUPER",
        name="Supertrend",
        category="trend",
        description="Trend following indicator based on ATR.",
        params=[
            FeatureParamSchema("length", "int", min=5, max=20, default=7),
            FeatureParamSchema("multiplier", "float", min=1.0, max=4.0, step=0.5, default=3.0),
        ]
    ),
    "ICHIMOKU": FeatureDefinition(
        id="ICHIMOKU",
        name="Ichimoku Cloud",
        category="trend",
        description="Collection of indicators showing support, resistance, and trend.",
        params=[
            FeatureParamSchema("tenkan", "int", min=5, max=20, default=9),
            FeatureParamSchema("kijun", "int", min=20, max=40, default=26),
            FeatureParamSchema("senkou", "int", min=40, max=60, default=52),
        ]
    ),

    # --- Volume ---
    "MFI": FeatureDefinition(
        id="MFI",
        name="Money Flow Index",
        category="volume",
        description="Volume-weighted RSI.",
        params=[
            FeatureParamSchema("length", "int", min=5, max=30, default=14),
        ]
    ),
    "CMF": FeatureDefinition(
        id="CMF",
        name="Chaikin Money Flow",
        category="volume",
        description="Measures money flow volume over a set period.",
        params=[
            FeatureParamSchema("length", "int", min=10, max=30, default=20),
        ]
    ),
    "EOM": FeatureDefinition(
        id="EOM",
        name="Ease of Movement",
        category="volume",
        description="Volume based oscillator showing how easily price can move.",
        params=[
            FeatureParamSchema("length", "int", min=5, max=30, default=14),
        ]
    ),

    # --- Volatility ---
    "KC": FeatureDefinition(
        id="KC",
        name="Keltner Channels",
        category="volatility",
        description="Volatility channels using ATR.",
        params=[
            FeatureParamSchema("length", "int", min=10, max=50, default=20),
            FeatureParamSchema("scalar", "float", min=1.0, max=3.0, step=0.1, default=2.0),
        ]
    ),
    "DONCHIAN": FeatureDefinition(
        id="DONCHIAN",
        name="Donchian Channels",
        category="volatility",
        description="Highest high and lowest low channels.",
        params=[
            FeatureParamSchema("lower_length", "int", min=10, max=50, default=20),
            FeatureParamSchema("upper_length", "int", min=10, max=50, default=20),
        ]
    ),
}
