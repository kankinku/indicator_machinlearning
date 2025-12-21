"""
Additional Feature Registration (Part 2)
Adds more indicators to reach the original ~50 feature count.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.registry import FeatureRegistry
from src.config import config
from src.contracts import FeatureMetadata, TunableParamSpec

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

FEATURES_EXTENDED = [
    # ==================== MOMENTUM (Extended) ====================
    (
        "MOMENTUM_AO_V1",
        "Awesome Oscillator",
        "momentum",
        "Bill Williams Awesome Oscillator",
        [],
        """
class AOHandler:
    def compute(self, df, **kwargs):
        from ta.momentum import AwesomeOscillatorIndicator
        ao = AwesomeOscillatorIndicator(high=df['high'], low=df['low'])
        result = pd.DataFrame(index=df.index)
        result['ao'] = ao.awesome_oscillator()
        return result.fillna(0)
""",
        "AOHandler"
    ),
    (
        "MOMENTUM_TSI_V1",
        "TSI",
        "momentum",
        "True Strength Index",
        [],
        """
class TSIHandler:
    def compute(self, df, **kwargs):
        from ta.momentum import TSIIndicator
        tsi = TSIIndicator(close=df['close'])
        result = pd.DataFrame(index=df.index)
        result['tsi'] = tsi.tsi()
        return result.fillna(0)
""",
        "TSIHandler"
    ),
    (
        "MOMENTUM_UO_V1",
        "Ultimate Oscillator",
        "momentum",
        "Ultimate Oscillator",
        [],
        """
class UOHandler:
    def compute(self, df, **kwargs):
        from ta.momentum import UltimateOscillator
        uo = UltimateOscillator(high=df['high'], low=df['low'], close=df['close'])
        result = pd.DataFrame(index=df.index)
        result['uo'] = uo.ultimate_oscillator()
        return result.fillna(0)
""",
        "UOHandler"
    ),
    (
        "MOMENTUM_KAMA_V1",
        "KAMA",
        "momentum",
        "Kaufman Adaptive Moving Average",
        [TunableParamSpec("window", "int", 5, 20, 1, 10)],
        """
class KAMAHandler:
    def compute(self, df, **kwargs):
        from ta.momentum import KAMAIndicator
        window = kwargs.get('window', 10)
        kama = KAMAIndicator(close=df['close'], window=window)
        result = pd.DataFrame(index=df.index)
        result['kama'] = kama.kama()
        return result.fillna(0)
""",
        "KAMAHandler"
    ),
    (
        "MOMENTUM_PPO_V1",
        "PPO",
        "momentum",
        "Percentage Price Oscillator",
        [],
        """
class PPOHandler:
    def compute(self, df, **kwargs):
        from ta.momentum import PercentagePriceOscillator
        ppo = PercentagePriceOscillator(close=df['close'])
        result = pd.DataFrame(index=df.index)
        result['ppo'] = ppo.ppo()
        result['ppo_signal'] = ppo.ppo_signal()
        result['ppo_hist'] = ppo.ppo_hist()
        return result.fillna(0)
""",
        "PPOHandler"
    ),
    (
        "MOMENTUM_PVO_V1",
        "PVO",
        "momentum",
        "Percentage Volume Oscillator",
        [],
        """
class PVOHandler:
    def compute(self, df, **kwargs):
        from ta.momentum import PercentageVolumeOscillator
        pvo = PercentageVolumeOscillator(volume=df['volume'])
        result = pd.DataFrame(index=df.index)
        result['pvo'] = pvo.pvo()
        result['pvo_signal'] = pvo.pvo_signal()
        result['pvo_hist'] = pvo.pvo_hist()
        return result.fillna(0)
""",
        "PVOHandler"
    ),
    
    # ==================== TREND (Extended) ====================
    (
        "TREND_TRIX_V1",
        "TRIX",
        "trend",
        "Triple Exponential Average",
        [TunableParamSpec("window", "int", 10, 20, 1, 15)],
        """
class TRIXHandler:
    def compute(self, df, **kwargs):
        from ta.trend import TRIXIndicator
        window = kwargs.get('window', 15)
        trix = TRIXIndicator(close=df['close'], window=window)
        result = pd.DataFrame(index=df.index)
        result['trix'] = trix.trix()
        return result.fillna(0)
""",
        "TRIXHandler"
    ),
    (
        "TREND_MASS_V1",
        "Mass Index",
        "trend",
        "Mass Index for reversal detection",
        [],
        """
class MassHandler:
    def compute(self, df, **kwargs):
        from ta.trend import MassIndex
        mi = MassIndex(high=df['high'], low=df['low'])
        result = pd.DataFrame(index=df.index)
        result['mass_index'] = mi.mass_index()
        return result.fillna(0)
""",
        "MassHandler"
    ),
    (
        "TREND_DPO_V1",
        "DPO",
        "trend",
        "Detrended Price Oscillator",
        [TunableParamSpec("window", "int", 10, 30, 1, 20)],
        """
class DPOHandler:
    def compute(self, df, **kwargs):
        from ta.trend import DPOIndicator
        window = kwargs.get('window', 20)
        dpo = DPOIndicator(close=df['close'], window=window)
        result = pd.DataFrame(index=df.index)
        result['dpo'] = dpo.dpo()
        return result.fillna(0)
""",
        "DPOHandler"
    ),
    (
        "TREND_KST_V1",
        "KST",
        "trend",
        "Know Sure Thing",
        [],
        """
class KSTHandler:
    def compute(self, df, **kwargs):
        from ta.trend import KSTIndicator
        kst = KSTIndicator(close=df['close'])
        result = pd.DataFrame(index=df.index)
        result['kst'] = kst.kst()
        result['kst_signal'] = kst.kst_sig()
        result['kst_diff'] = kst.kst_diff()
        return result.fillna(0)
""",
        "KSTHandler"
    ),
    (
        "TREND_STC_V1",
        "STC",
        "trend",
        "Schaff Trend Cycle",
        [],
        """
class STCHandler:
    def compute(self, df, **kwargs):
        from ta.trend import STCIndicator
        stc = STCIndicator(close=df['close'])
        result = pd.DataFrame(index=df.index)
        result['stc'] = stc.stc()
        return result.fillna(0)
""",
        "STCHandler"
    ),
    (
        "TREND_VORTEX_V1",
        "Vortex",
        "trend",
        "Vortex Indicator",
        [TunableParamSpec("window", "int", 7, 21, 1, 14)],
        """
class VortexHandler:
    def compute(self, df, **kwargs):
        from ta.trend import VortexIndicator
        window = kwargs.get('window', 14)
        vi = VortexIndicator(high=df['high'], low=df['low'], close=df['close'], window=window)
        result = pd.DataFrame(index=df.index)
        result['vortex_pos'] = vi.vortex_indicator_pos()
        result['vortex_neg'] = vi.vortex_indicator_neg()
        result['vortex_diff'] = vi.vortex_indicator_diff()
        return result.fillna(0)
""",
        "VortexHandler"
    ),
    (
        "TREND_PSAR_V1",
        "PSAR",
        "trend",
        "Parabolic SAR",
        [],
        """
class PSARHandler:
    def compute(self, df, **kwargs):
        import warnings
        warnings.filterwarnings('ignore')
        from ta.trend import PSARIndicator
        psar = PSARIndicator(high=df['high'], low=df['low'], close=df['close'])
        result = pd.DataFrame(index=df.index)
        result['psar'] = psar.psar()
        result['psar_up'] = psar.psar_up()
        result['psar_down'] = psar.psar_down()
        return result.fillna(0)
""",
        "PSARHandler"
    ),
    (
        "TREND_WMA_V1",
        "WMA",
        "trend",
        "Weighted Moving Average",
        [TunableParamSpec("window", "int", 5, 50, 5, 20)],
        """
class WMAHandler:
    def compute(self, df, **kwargs):
        from ta.trend import WMAIndicator
        window = kwargs.get('window', 20)
        wma = WMAIndicator(close=df['close'], window=window)
        result = pd.DataFrame(index=df.index)
        result['wma'] = wma.wma()
        return result.fillna(0)
""",
        "WMAHandler"
    ),
    
    # ==================== VOLATILITY (Extended) ====================
    (
        "VOLATILITY_DC_V1",
        "Donchian Channel",
        "volatility",
        "Donchian Channel",
        [TunableParamSpec("window", "int", 10, 30, 1, 20)],
        """
class DCHandler:
    def compute(self, df, **kwargs):
        from ta.volatility import DonchianChannel
        window = kwargs.get('window', 20)
        dc = DonchianChannel(high=df['high'], low=df['low'], close=df['close'], window=window)
        result = pd.DataFrame(index=df.index)
        result['dc_high'] = dc.donchian_channel_hband()
        result['dc_low'] = dc.donchian_channel_lband()
        result['dc_mid'] = dc.donchian_channel_mband()
        result['dc_width'] = dc.donchian_channel_wband()
        return result.fillna(0)
""",
        "DCHandler"
    ),
    (
        "VOLATILITY_UI_V1",
        "Ulcer Index",
        "volatility",
        "Ulcer Index",
        [TunableParamSpec("window", "int", 7, 21, 1, 14)],
        """
class UIHandler:
    def compute(self, df, **kwargs):
        from ta.volatility import UlcerIndex
        window = kwargs.get('window', 14)
        ui = UlcerIndex(close=df['close'], window=window)
        result = pd.DataFrame(index=df.index)
        result['ulcer_index'] = ui.ulcer_index()
        return result.fillna(0)
""",
        "UIHandler"
    ),
    
    # ==================== VOLUME (Extended) ====================
    (
        "VOLUME_FI_V1",
        "Force Index",
        "volume",
        "Elder's Force Index",
        [TunableParamSpec("window", "int", 7, 21, 1, 13)],
        """
class FIHandler:
    def compute(self, df, **kwargs):
        from ta.volume import ForceIndexIndicator
        window = kwargs.get('window', 13)
        fi = ForceIndexIndicator(close=df['close'], volume=df['volume'], window=window)
        result = pd.DataFrame(index=df.index)
        result['force_index'] = fi.force_index()
        return result.fillna(0)
""",
        "FIHandler"
    ),
    (
        "VOLUME_EOM_V1",
        "EOM",
        "volume",
        "Ease of Movement",
        [TunableParamSpec("window", "int", 7, 21, 1, 14)],
        """
class EOMHandler:
    def compute(self, df, **kwargs):
        from ta.volume import EaseOfMovementIndicator
        window = kwargs.get('window', 14)
        eom = EaseOfMovementIndicator(high=df['high'], low=df['low'], volume=df['volume'], window=window)
        result = pd.DataFrame(index=df.index)
        result['eom'] = eom.ease_of_movement()
        result['eom_sma'] = eom.sma_ease_of_movement()
        return result.fillna(0)
""",
        "EOMHandler"
    ),
    (
        "VOLUME_VPT_V1",
        "VPT",
        "volume",
        "Volume Price Trend",
        [],
        """
class VPTHandler:
    def compute(self, df, **kwargs):
        from ta.volume import VolumePriceTrendIndicator
        vpt = VolumePriceTrendIndicator(close=df['close'], volume=df['volume'])
        result = pd.DataFrame(index=df.index)
        result['vpt'] = vpt.volume_price_trend()
        return result.fillna(0)
""",
        "VPTHandler"
    ),
    (
        "VOLUME_NVI_V1",
        "NVI",
        "volume",
        "Negative Volume Index",
        [],
        """
class NVIHandler:
    def compute(self, df, **kwargs):
        from ta.volume import NegativeVolumeIndexIndicator
        nvi = NegativeVolumeIndexIndicator(close=df['close'], volume=df['volume'])
        result = pd.DataFrame(index=df.index)
        result['nvi'] = nvi.negative_volume_index()
        return result.fillna(0)
""",
        "NVIHandler"
    ),
    (
        "VOLUME_VWAP_V1",
        "VWAP",
        "volume",
        "Volume Weighted Average Price",
        [TunableParamSpec("window", "int", 7, 21, 1, 14)],
        """
class VWAPHandler:
    def compute(self, df, **kwargs):
        from ta.volume import VolumeWeightedAveragePrice
        window = kwargs.get('window', 14)
        vwap = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=window)
        result = pd.DataFrame(index=df.index)
        result['vwap'] = vwap.volume_weighted_average_price()
        return result.fillna(0)
""",
        "VWAPHandler"
    ),
    
    # ==================== CUSTOM / DERIVED ====================
    (
        "CUSTOM_RETURN_V1",
        "Returns",
        "custom",
        "Price Returns (1, 5, 20 day)",
        [],
        """
class ReturnHandler:
    def compute(self, df, **kwargs):
        result = pd.DataFrame(index=df.index)
        result['ret_1d'] = df['close'].pct_change(1)
        result['ret_5d'] = df['close'].pct_change(5)
        result['ret_20d'] = df['close'].pct_change(20)
        return result.fillna(0)
""",
        "ReturnHandler"
    ),
    (
        "CUSTOM_VOLATILITY_V1",
        "Realized Volatility",
        "custom",
        "Rolling Realized Volatility",
        [TunableParamSpec("window", "int", 5, 30, 5, 20)],
        """
class RealizedVolHandler:
    def compute(self, df, **kwargs):
        window = kwargs.get('window', 20)
        returns = df['close'].pct_change()
        result = pd.DataFrame(index=df.index)
        result['rvol'] = returns.rolling(window=window).std() * np.sqrt(252)
        return result.fillna(0)
""",
        "RealizedVolHandler"
    ),
    (
        "CUSTOM_RANGE_V1",
        "Price Range",
        "custom",
        "High-Low Range as % of Close",
        [],
        """
class RangeHandler:
    def compute(self, df, **kwargs):
        result = pd.DataFrame(index=df.index)
        result['range_pct'] = (df['high'] - df['low']) / df['close']
        result['range_20d_avg'] = result['range_pct'].rolling(20).mean()
        return result.fillna(0)
""",
        "RangeHandler"
    ),
    (
        "CUSTOM_GAP_V1",
        "Gap",
        "custom",
        "Opening Gap from Previous Close",
        [],
        """
class GapHandler:
    def compute(self, df, **kwargs):
        result = pd.DataFrame(index=df.index)
        result['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        result['gap_up'] = (result['gap'] > 0.01).astype(int)
        result['gap_down'] = (result['gap'] < -0.01).astype(int)
        return result.fillna(0)
""",
        "GapHandler"
    ),
    (
        "CUSTOM_MOMENTUM_RANK_V1",
        "Momentum Rank",
        "custom",
        "Percentile Rank of Momentum",
        [TunableParamSpec("window", "int", 20, 60, 10, 60)],
        """
class MomentumRankHandler:
    def compute(self, df, **kwargs):
        window = kwargs.get('window', 60)
        result = pd.DataFrame(index=df.index)
        mom = df['close'].pct_change(20)
        result['mom_rank'] = mom.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        return result.fillna(0.5)
""",
        "MomentumRankHandler"
    ),
    
    # ==================== CONTEXT (Time, Stats, Relative) ====================
    (
        "CONTEXT_TIME_V1",
        "Time Features",
        "context",
        "Cyclical Time Encoding (Day, Month)",
        [],
        """
class TimeHandler:
    def compute(self, df, **kwargs):
        result = pd.DataFrame(index=df.index)
        if hasattr(df.index, 'dayofweek'):
            dow = df.index.dayofweek
            month = df.index.month
        else:
            dow = pd.to_datetime(df.index).dayofweek
            month = pd.to_datetime(df.index).month
        result['time_dow_sin'] = np.sin(2 * np.pi * dow / 7)
        result['time_dow_cos'] = np.cos(2 * np.pi * dow / 7)
        result['time_month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
        result['time_month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
        return result.fillna(0)
""",
        "TimeHandler"
    ),
    (
        "CONTEXT_STAT_V1",
        "Statistical Moments",
        "context",
        "Skewness and Kurtosis",
        [TunableParamSpec("window", "int", 10, 60, 10, 20)],
        """
class StatHandler:
    def compute(self, df, **kwargs):
        window = kwargs.get('window', 20)
        result = pd.DataFrame(index=df.index)
        result['skew'] = df['close'].rolling(window).skew()
        result['kurt'] = df['close'].rolling(window).kurt()
        return result.fillna(0)
""",
        "StatHandler"
    ),
    (
        "CUSTOM_ZSCORE_V1",
        "Z-Score",
        "custom",
        "Price Z-Score relative to rolling window",
        [TunableParamSpec("window", "int", 10, 60, 10, 20)],
        """
class ZScoreHandler:
    def compute(self, df, **kwargs):
        window = kwargs.get('window', 20)
        result = pd.DataFrame(index=df.index)
        mean = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        result['zscore'] = (df['close'] - mean) / (std + 1e-9)
        return result.fillna(0)
""",
        "ZScoreHandler"
    ),
]

def test_snippet(snippet, handler_name, df):
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
    print(">>> [PopulateV2 Extended] Adding more indicators...")
    
    test_df = create_test_df()
    registry = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH))
    registry.initialize()
    
    success_count = 0
    fail_count = 0
    
    for f_id, name, category, description, params, snippet, handler in FEATURES_EXTENDED:
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
    
    print(f"\n>>> [PopulateV2 Extended] Done. Added: {success_count}, Failed: {fail_count}")
    print(f"    Total features in registry: {len(registry._features)}")

if __name__ == "__main__":
    main()
