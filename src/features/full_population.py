
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import config
from src.features.registry import FeatureRegistry
from src.contracts import FeatureMetadata, TunableParamSpec

def populate_full_population():
    """
    Populates ALL remaining indicators from definitions.py into the registry.
    This completes the migration to the Dynamic Registry.
    """
    registry = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH))
    registry.initialize()
    
    print(f"Populating ALL remaining features into {config.FEATURE_REGISTRY_PATH}...")
    
    # 9. ROC (Rate of Change)
    roc_code = """
import pandas as pd
import ta

class ROCIndicator:
    def compute(self, df: pd.DataFrame, length: int = 12) -> pd.DataFrame:
        vals = ta.momentum.ROCIndicator(df["close"], window=length).roc()
        return pd.DataFrame({f"ROC_{length}": vals}, index=df.index)
"""
    registry.register(FeatureMetadata(
        feature_id="MOMENTUM_ROC_V1",
        name="Rate of Change",
        category="MOMENTUM",
        description="Pure momentum oscillator.",
        code_snippet=roc_code,
        handler_func="ROCIndicator",
        params=[TunableParamSpec(name="length", param_type="int", min=1, max=50, default=12)],
        source="builtin"
    ))

    # 10. WILLR (Williams %R)
    willr_code = """
import pandas as pd
import ta

class WillRIndicator:
    def compute(self, df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        vals = ta.momentum.WilliamsRIndicator(high=df["high"], low=df["low"], close=df["close"], lbp=length).williams_r()
        return pd.DataFrame({f"WILLR_{length}": vals}, index=df.index)
"""
    registry.register(FeatureMetadata(
        feature_id="MOMENTUM_WILLR_V1",
        name="Williams %R",
        category="MOMENTUM",
        description="Momentum indicator moving between 0 and -100.",
        code_snippet=willr_code,
        handler_func="WillRIndicator",
        params=[TunableParamSpec(name="length", param_type="int", min=5, max=50, default=14)],
        source="builtin"
    ))

    # 11. MOM (Simple Momentum)
    mom_code = """
import pandas as pd

class MomIndicator:
    def compute(self, df: pd.DataFrame, length: int = 10) -> pd.DataFrame:
        mom = df["close"].diff(periods=length)
        return pd.DataFrame({f"MOM_{length}": mom}, index=df.index)
"""
    registry.register(FeatureMetadata(
        feature_id="MOMENTUM_MOM_V1",
        name="Momentum (Simple)",
        category="MOMENTUM",
        description="Simple price change.",
        code_snippet=mom_code,
        handler_func="MomIndicator",
        params=[TunableParamSpec(name="length", param_type="int", min=1, max=50, default=10)],
        source="builtin"
    ))

    # 12. CMO (Chande Momentum Oscillator)
    cmo_code = """
import pandas as pd
import numpy as np

class CMOIndicator:
    def compute(self, df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        mom = df["close"].diff()
        up = mom.clip(lower=0)
        down = -mom.clip(upper=0)
        su = up.rolling(length).sum()
        sd = down.rolling(length).sum()
        cmo = 100 * (su - sd) / (su + sd + 1e-9)
        return pd.DataFrame({f"CMO_{length}": cmo}, index=df.index)
"""
    registry.register(FeatureMetadata(
        feature_id="MOMENTUM_CMO_V1",
        name="Chande Momentum Oscillator",
        category="MOMENTUM",
        description="Chande Momentum Oscillator.",
        code_snippet=cmo_code,
        handler_func="CMOIndicator",
        params=[TunableParamSpec(name="length", param_type="int", min=5, max=50, default=14)],
        source="builtin"
    ))

    # 13. ADX
    adx_code = """
import pandas as pd
import ta

class ADXIndicator:
    def compute(self, df: pd.DataFrame, length: int = 14, lensig: int = 14) -> pd.DataFrame:
        # Note: ta lib adx uses one window usually, or separates ADX smoothing.
        # We pass length as window.
        adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=length)
        return pd.DataFrame({
            f"ADX_{length}": adx_ind.adx(),
            f"ADX_pos_{length}": adx_ind.adx_pos(),
            f"ADX_neg_{length}": adx_ind.adx_neg()
        }, index=df.index)
"""
    registry.register(FeatureMetadata(
        feature_id="TREND_ADX_V1",
        name="Average Directional Index",
        category="TREND",
        description="Trend strength.",
        code_snippet=adx_code,
        handler_func="ADXIndicator",
        params=[
            TunableParamSpec(name="length", param_type="int", min=5, max=50, default=14),
            TunableParamSpec(name="lensig", param_type="int", min=5, max=50, default=14)
        ],
        source="builtin"
    ))

    # 14. TRIX
    trix_code = """
import pandas as pd
import ta

class TRIXIndicator:
    def compute(self, df: pd.DataFrame, length: int = 15, signal: int = 9) -> pd.DataFrame:
        trix = ta.trend.TRIXIndicator(df["close"], window=length).trix()
        return pd.DataFrame({f"TRIX_{length}": trix}, index=df.index)
"""
    registry.register(FeatureMetadata(
        feature_id="TREND_TRIX_V1",
        name="TRIX",
        category="TREND",
        description="Triple Exponential Average.",
        code_snippet=trix_code,
        handler_func="TRIXIndicator",
        params=[
            TunableParamSpec(name="length", param_type="int", min=5, max=50, default=15),
            TunableParamSpec(name="signal", param_type="int", min=5, max=50, default=9)
        ],
        source="builtin"
    ))

    # 15. Supertrend (SUPER)
    super_code = """
import pandas as pd
import ta

class SuperTrendIndicator:
    def compute(self, df: pd.DataFrame, length: int = 7, multiplier: float = 3.0) -> pd.DataFrame:
        atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=length).average_true_range()
        hl2 = (df["high"] + df["low"]) / 2
        
        up = hl2 + multiplier * atr
        down = hl2 - multiplier * atr
        
        dist_up = (df["close"] - up) / df["close"]
        dist_down = (df["close"] - down) / df["close"]
        
        return pd.DataFrame({
            f"SUPER_dist_up_{length}": dist_up,
            f"SUPER_dist_down_{length}": dist_down
        }, index=df.index)
"""
    registry.register(FeatureMetadata(
        feature_id="TREND_SUPER_V1",
        name="Supertrend",
        category="TREND",
        description="Supertrend distance metrics.",
        code_snippet=super_code,
        handler_func="SuperTrendIndicator",
        params=[
            TunableParamSpec(name="length", param_type="int", min=5, max=50, default=7),
            TunableParamSpec(name="multiplier", param_type="float", min=1.0, max=5.0, default=3.0)
        ],
        source="builtin"
    ))

    # 16. Ichimoku
    ichi_code = """
import pandas as pd
import ta

class IchimokuIndicator:
    def compute(self, df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
        ich = ta.trend.IchimokuIndicator(df["high"], df["low"], window1=tenkan, window2=kijun, window3=senkou)
        return pd.DataFrame({
            f"ICHI_conv_{tenkan}": ich.ichimoku_conversion_line(),
            f"ICHI_base_{kijun}": ich.ichimoku_base_line(),
            f"ICHI_a_{senkou}": ich.ichimoku_a(),
            f"ICHI_b_{senkou}": ich.ichimoku_b()
        }, index=df.index)
"""
    registry.register(FeatureMetadata(
        feature_id="TREND_ICHIMOKU_V1",
        name="Ichimoku Cloud",
        category="TREND",
        description="Ichimoku Cloud components.",
        code_snippet=ichi_code,
        handler_func="IchimokuIndicator",
        params=[
            TunableParamSpec(name="tenkan", param_type="int", min=5, max=20, default=9),
            TunableParamSpec(name="kijun", param_type="int", min=20, max=40, default=26),
            TunableParamSpec(name="senkou", param_type="int", min=40, max=80, default=52)
        ],
        source="builtin"
    ))

    # 17. MFI
    mfi_code = """
import pandas as pd
import ta

class MFIIndicator:
    def compute(self, df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        mfi = ta.volume.MFIIndicator(df["high"], df["low"], df["close"], df["volume"], window=length).money_flow_index()
        return pd.DataFrame({f"MFI_{length}": mfi}, index=df.index)
"""
    registry.register(FeatureMetadata(
        feature_id="VOLUME_MFI_V1",
        name="Money Flow Index",
        category="VOLUME",
        description="Money Flow Index.",
        code_snippet=mfi_code,
        handler_func="MFIIndicator",
        params=[TunableParamSpec(name="length", param_type="int", min=5, max=50, default=14)],
        source="builtin"
    ))

    # 18. CMF
    cmf_code = """
import pandas as pd
import ta

class CMFIndicator:
    def compute(self, df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
        cmf = ta.volume.ChaikinMoneyFlowIndicator(df["high"], df["low"], df["close"], df["volume"], window=length).chaikin_money_flow()
        return pd.DataFrame({f"CMF_{length}": cmf}, index=df.index)
"""
    registry.register(FeatureMetadata(
        feature_id="VOLUME_CMF_V1",
        name="Chaikin Money Flow",
        category="VOLUME",
        description="Chaikin Money Flow.",
        code_snippet=cmf_code,
        handler_func="CMFIndicator",
        params=[TunableParamSpec(name="length", param_type="int", min=10, max=50, default=20)],
        source="builtin"
    ))

    # 19. EOM
    eom_code = """
import pandas as pd
import ta

class EOMIndicator:
    def compute(self, df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        eom = ta.volume.EaseOfMovementIndicator(df["high"], df["low"], df["volume"], window=length).ease_of_movement()
        return pd.DataFrame({f"EOM_{length}": eom}, index=df.index)
"""
    registry.register(FeatureMetadata(
        feature_id="VOLUME_EOM_V1",
        name="Ease of Movement",
        category="VOLUME",
        description="Ease of Movement.",
        code_snippet=eom_code,
        handler_func="EOMIndicator",
        params=[TunableParamSpec(name="length", param_type="int", min=5, max=50, default=14)],
        source="builtin"
    ))

    # 20. KC (Keltner Channels)
    kc_code = """
import pandas as pd
import ta

class KCIndicator:
    def compute(self, df: pd.DataFrame, length: int = 20, scalar: float = 2.0) -> pd.DataFrame:
        kc = ta.volatility.KeltnerChannel(df["high"], df["low"], df["close"], window=length, window_atr=length)
        return pd.DataFrame({
            f"KC_h_{length}": kc.keltner_channel_hband(),
            f"KC_l_{length}": kc.keltner_channel_lband(),
            f"KC_w_{length}": kc.keltner_channel_wband()
        }, index=df.index)
"""
    registry.register(FeatureMetadata(
        feature_id="VOLATILITY_KC_V1",
        name="Keltner Channels",
        category="VOLATILITY",
        description="Keltner Channels.",
        code_snippet=kc_code,
        handler_func="KCIndicator",
        params=[
            TunableParamSpec(name="length", param_type="int", min=10, max=50, default=20),
            TunableParamSpec(name="scalar", param_type="float", min=1.0, max=4.0, default=2.0)
        ],
        source="builtin"
    ))

    # 21. Donchian Channels
    dc_code = """
import pandas as pd
import ta

class DCIndicator:
    def compute(self, df: pd.DataFrame, upper_length: int = 20, lower_length: int = 20) -> pd.DataFrame:
        # Note: ta lib donchian implies symmetric usually, or we use max. We just use upper_length as window.
        dc = ta.volatility.DonchianChannel(df["high"], df["low"], df["close"], window=upper_length)
        return pd.DataFrame({
            f"DC_h_{upper_length}": dc.donchian_channel_hband(),
            f"DC_l_{upper_length}": dc.donchian_channel_lband(),
            f"DC_w_{upper_length}": dc.donchian_channel_wband()
        }, index=df.index)
"""
    registry.register(FeatureMetadata(
        feature_id="VOLATILITY_DONCHIAN_V1",
        name="Donchian Channels",
        category="VOLATILITY",
        description="Donchian High/Low Channels.",
        code_snippet=dc_code,
        handler_func="DCIndicator",
        params=[
            TunableParamSpec(name="upper_length", param_type="int", min=10, max=50, default=20),
            TunableParamSpec(name="lower_length", param_type="int", min=10, max=50, default=20)
        ],
        source="builtin"
    ))

    print(f"Success! Full population (21 indicators) populated.")

if __name__ == "__main__":
    populate_full_population()
