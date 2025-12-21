"""
Feature Registry Population Script V2 (2024-12-21)

This script populates the feature registry with TESTED and WORKING indicators.
Each indicator snippet is validated against dummy OHLCV data before registration.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.registry import FeatureRegistry
from src.config import config
from src.contracts import FeatureMetadata, TunableParamSpec

# --- Test Data ---
def create_test_df(n=200):
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='D')
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    df = pd.DataFrame({
        'open': close * (1 + np.random.randn(n) * 0.01),
        'high': close * (1 + abs(np.random.randn(n) * 0.02)),
        'low': close * (1 - abs(np.random.randn(n) * 0.02)),
        'close': close,
        'volume': np.random.randint(100000, 1000000, n).astype(float)
    }, index=dates)
    return df

# --- Core Feature Definitions ---
# Each entry: (feature_id, name, category, description, params, code_snippet, handler_func)

FEATURES = [
    # ==================== MOMENTUM ====================
    (
        "MOMENTUM_RSI_V1",
        "RSI",
        "momentum",
        "Relative Strength Index",
        [TunableParamSpec("window", "int", 5, 30, 1, 14)],
        """
class RSIHandler:
    def compute(self, df, **kwargs):
        from ta.momentum import RSIIndicator
        window = kwargs.get('window', 14)
        rsi = RSIIndicator(close=df['close'], window=window)
        result = pd.DataFrame(index=df.index)
        result['rsi'] = rsi.rsi()
        return result.fillna(0)
""",
        "RSIHandler"
    ),
    (
        "MOMENTUM_MACD_V1",
        "MACD",
        "momentum",
        "Moving Average Convergence Divergence",
        [
            TunableParamSpec("fast", "int", 8, 15, 1, 12),
            TunableParamSpec("slow", "int", 20, 30, 1, 26),
            TunableParamSpec("signal", "int", 5, 12, 1, 9),
        ],
        """
class MACDHandler:
    def compute(self, df, **kwargs):
        from ta.trend import MACD
        fast = kwargs.get('fast', 12)
        slow = kwargs.get('slow', 26)
        signal = kwargs.get('signal', 9)
        macd = MACD(close=df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
        result = pd.DataFrame(index=df.index)
        result['macd'] = macd.macd()
        result['macd_signal'] = macd.macd_signal()
        result['macd_diff'] = macd.macd_diff()
        return result.fillna(0)
""",
        "MACDHandler"
    ),
    (
        "MOMENTUM_ROC_V1",
        "ROC",
        "momentum",
        "Rate of Change",
        [TunableParamSpec("window", "int", 5, 20, 1, 12)],
        """
class ROCHandler:
    def compute(self, df, **kwargs):
        from ta.momentum import ROCIndicator
        window = kwargs.get('window', 12)
        roc = ROCIndicator(close=df['close'], window=window)
        result = pd.DataFrame(index=df.index)
        result['roc'] = roc.roc()
        return result.fillna(0)
""",
        "ROCHandler"
    ),
    (
        "MOMENTUM_STOCH_V1",
        "Stochastic",
        "momentum",
        "Stochastic Oscillator",
        [TunableParamSpec("window", "int", 5, 21, 1, 14)],
        """
class StochHandler:
    def compute(self, df, **kwargs):
        from ta.momentum import StochasticOscillator
        window = kwargs.get('window', 14)
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=window)
        result = pd.DataFrame(index=df.index)
        result['stoch_k'] = stoch.stoch()
        result['stoch_d'] = stoch.stoch_signal()
        return result.fillna(0)
""",
        "StochHandler"
    ),
    (
        "MOMENTUM_CCI_V1",
        "CCI",
        "momentum",
        "Commodity Channel Index",
        [TunableParamSpec("window", "int", 10, 30, 1, 20)],
        """
class CCIHandler:
    def compute(self, df, **kwargs):
        from ta.trend import CCIIndicator
        window = kwargs.get('window', 20)
        cci = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=window)
        result = pd.DataFrame(index=df.index)
        result['cci'] = cci.cci()
        return result.fillna(0)
""",
        "CCIHandler"
    ),
    (
        "MOMENTUM_WILLR_V1",
        "Williams %R",
        "momentum",
        "Williams Percent Range",
        [TunableParamSpec("window", "int", 7, 21, 1, 14)],
        """
class WillRHandler:
    def compute(self, df, **kwargs):
        from ta.momentum import WilliamsRIndicator
        window = kwargs.get('window', 14)
        wr = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=window)
        result = pd.DataFrame(index=df.index)
        result['willr'] = wr.williams_r()
        return result.fillna(0)
""",
        "WillRHandler"
    ),

    # ==================== TREND ====================
    (
        "TREND_SMA_V1",
        "SMA",
        "trend",
        "Simple Moving Average",
        [TunableParamSpec("window", "int", 5, 50, 5, 20)],
        """
class SMAHandler:
    def compute(self, df, **kwargs):
        window = kwargs.get('window', 20)
        result = pd.DataFrame(index=df.index)
        result['sma'] = df['close'].rolling(window=window).mean()
        return result.fillna(0)
""",
        "SMAHandler"
    ),
    (
        "TREND_EMA_V1",
        "EMA",
        "trend",
        "Exponential Moving Average",
        [TunableParamSpec("window", "int", 5, 50, 5, 20)],
        """
class EMAHandler:
    def compute(self, df, **kwargs):
        window = kwargs.get('window', 20)
        result = pd.DataFrame(index=df.index)
        result['ema'] = df['close'].ewm(span=window, adjust=False).mean()
        return result.fillna(0)
""",
        "EMAHandler"
    ),
    (
        "TREND_ADX_V1",
        "ADX",
        "trend",
        "Average Directional Index",
        [TunableParamSpec("window", "int", 7, 21, 1, 14)],
        """
class ADXHandler:
    def compute(self, df, **kwargs):
        from ta.trend import ADXIndicator
        window = kwargs.get('window', 14)
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=window)
        result = pd.DataFrame(index=df.index)
        result['adx'] = adx.adx()
        result['adx_pos'] = adx.adx_pos()
        result['adx_neg'] = adx.adx_neg()
        return result.fillna(0)
""",
        "ADXHandler"
    ),
    (
        "TREND_AROON_V1",
        "Aroon",
        "trend",
        "Aroon Indicator",
        [TunableParamSpec("window", "int", 10, 30, 5, 25)],
        """
class AroonHandler:
    def compute(self, df, **kwargs):
        from ta.trend import AroonIndicator
        window = kwargs.get('window', 25)
        aroon = AroonIndicator(high=df['high'], low=df['low'], window=window)
        result = pd.DataFrame(index=df.index)
        result['aroon_up'] = aroon.aroon_up()
        result['aroon_down'] = aroon.aroon_down()
        result['aroon_ind'] = aroon.aroon_indicator()
        return result.fillna(0)
""",
        "AroonHandler"
    ),
    (
        "TREND_ICHIMOKU_V1",
        "Ichimoku",
        "trend",
        "Ichimoku Cloud",
        [],
        """
class IchimokuHandler:
    def compute(self, df, **kwargs):
        from ta.trend import IchimokuIndicator
        ich = IchimokuIndicator(high=df['high'], low=df['low'])
        result = pd.DataFrame(index=df.index)
        result['ich_a'] = ich.ichimoku_a()
        result['ich_b'] = ich.ichimoku_b()
        result['ich_base'] = ich.ichimoku_base_line()
        result['ich_conv'] = ich.ichimoku_conversion_line()
        return result.fillna(0)
""",
        "IchimokuHandler"
    ),
    
    # ==================== VOLATILITY ====================
    (
        "VOLATILITY_ATR_V1",
        "ATR",
        "volatility",
        "Average True Range",
        [TunableParamSpec("window", "int", 7, 21, 1, 14)],
        """
class ATRHandler:
    def compute(self, df, **kwargs):
        from ta.volatility import AverageTrueRange
        window = kwargs.get('window', 14)
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=window)
        result = pd.DataFrame(index=df.index)
        result['atr'] = atr.average_true_range()
        return result.fillna(0)
""",
        "ATRHandler"
    ),
    (
        "VOLATILITY_BB_V1",
        "Bollinger Bands",
        "volatility",
        "Bollinger Bands Width and Position",
        [
            TunableParamSpec("window", "int", 10, 30, 1, 20),
            TunableParamSpec("std", "float", 1.5, 3.0, 0.5, 2.0),
        ],
        """
class BBHandler:
    def compute(self, df, **kwargs):
        from ta.volatility import BollingerBands
        window = kwargs.get('window', 20)
        std = kwargs.get('std', 2.0)
        bb = BollingerBands(close=df['close'], window=window, window_dev=std)
        result = pd.DataFrame(index=df.index)
        result['bb_high'] = bb.bollinger_hband()
        result['bb_low'] = bb.bollinger_lband()
        result['bb_mid'] = bb.bollinger_mavg()
        result['bb_width'] = bb.bollinger_wband()
        result['bb_pband'] = bb.bollinger_pband()
        return result.fillna(0)
""",
        "BBHandler"
    ),
    (
        "VOLATILITY_KC_V1",
        "Keltner Channel",
        "volatility",
        "Keltner Channel",
        [TunableParamSpec("window", "int", 10, 30, 1, 20)],
        """
class KCHandler:
    def compute(self, df, **kwargs):
        from ta.volatility import KeltnerChannel
        window = kwargs.get('window', 20)
        kc = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=window)
        result = pd.DataFrame(index=df.index)
        result['kc_high'] = kc.keltner_channel_hband()
        result['kc_low'] = kc.keltner_channel_lband()
        result['kc_mid'] = kc.keltner_channel_mband()
        result['kc_width'] = kc.keltner_channel_wband()
        return result.fillna(0)
""",
        "KCHandler"
    ),
    
    # ==================== VOLUME ====================
    (
        "VOLUME_OBV_V1",
        "OBV",
        "volume",
        "On-Balance Volume",
        [],
        """
class OBVHandler:
    def compute(self, df, **kwargs):
        from ta.volume import OnBalanceVolumeIndicator
        obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        result = pd.DataFrame(index=df.index)
        result['obv'] = obv.on_balance_volume()
        return result.fillna(0)
""",
        "OBVHandler"
    ),
    (
        "VOLUME_MFI_V1",
        "MFI",
        "volume",
        "Money Flow Index",
        [TunableParamSpec("window", "int", 7, 21, 1, 14)],
        """
class MFIHandler:
    def compute(self, df, **kwargs):
        from ta.volume import MFIIndicator
        window = kwargs.get('window', 14)
        mfi = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=window)
        result = pd.DataFrame(index=df.index)
        result['mfi'] = mfi.money_flow_index()
        return result.fillna(0)
""",
        "MFIHandler"
    ),
    (
        "VOLUME_CMF_V1",
        "CMF",
        "volume",
        "Chaikin Money Flow",
        [TunableParamSpec("window", "int", 10, 30, 1, 20)],
        """
class CMFHandler:
    def compute(self, df, **kwargs):
        from ta.volume import ChaikinMoneyFlowIndicator
        window = kwargs.get('window', 20)
        cmf = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=window)
        result = pd.DataFrame(index=df.index)
        result['cmf'] = cmf.chaikin_money_flow()
        return result.fillna(0)
""",
        "CMFHandler"
    ),
    (
        "VOLUME_ADL_V1",
        "ADL",
        "volume",
        "Accumulation/Distribution Line",
        [],
        """
class ADLHandler:
    def compute(self, df, **kwargs):
        from ta.volume import AccDistIndexIndicator
        adl = AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
        result = pd.DataFrame(index=df.index)
        result['adl'] = adl.acc_dist_index()
        return result.fillna(0)
""",
        "ADLHandler"
    ),
]

def test_snippet(snippet, handler_name, df):
    """Execute the snippet and test if it works."""
    try:
        exec_globals = {"pd": pd, "np": np}
        exec(snippet, exec_globals)
        handler_class = exec_globals.get(handler_name)
        if handler_class is None:
            return False, f"Handler {handler_name} not found"
        
        instance = handler_class()
        result = instance.compute(df)
        if result is None or (isinstance(result, pd.DataFrame) and result.empty):
            return False, "Empty result"
        return True, "OK"
    except Exception as e:
        return False, str(e)

def main():
    print(">>> [PopulateV2] Creating feature registry with tested indicators...")
    
    # Prepare test data
    test_df = create_test_df()
    
    # Load registry
    registry = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH))
    registry.initialize()
    
    success_count = 0
    fail_count = 0
    
    for f_id, name, category, description, params, snippet, handler in FEATURES:
        print(f"    Testing {f_id}...", end=" ")
        
        ok, msg = test_snippet(snippet, handler, test_df)
        
        if ok:
            print("PASS -> Registering")
            meta = FeatureMetadata(
                feature_id=f_id,
                name=name,
                category=category,
                description=description,
                params=params,
                code_snippet=snippet,
                handler_func=handler
            )
            registry.register(meta, overwrite=True)
            success_count += 1
        else:
            print(f"FAIL ({msg})")
            fail_count += 1
    
    print(f"\n>>> [PopulateV2] Done. Registered: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()
