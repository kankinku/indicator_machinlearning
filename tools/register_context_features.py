
import sys
import os
from pathlib import Path

# Setup Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.registry import FeatureRegistry
from src.config import config
from src.contracts import FeatureMetadata, FeatureParam

def register_context_features():
    registry = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH))
    registry.initialize()
    
    # 1. Macro Data Wrappers
    # These snippets rely on the fact that 'df' passed to factory ALREADY has these columns from DataLoader
    macro_map = {
        "VIX": "vix_close",
        "US10Y": "^tnx_close", # Lowercased from ^TNX -> ^tnx_close
        "DX": "dx-y.nyb_close",
        "BTC": "btc-usd_close",
        "USO": "uso_close",
        "HYG": "hyg_close"
    }
    
    for key, col_name in macro_map.items():
        # Clean col name for python access if needed, but dict access is fine
        # Snippet logic: Just selecting the column
        snippet = f"""
class Macro{key}Handler:
    def compute(self, df, **kwargs):
        # DF columns are lowercased in factory
        # Look for partial matches or specific mapped names
        target = "{col_name}".lower()
        if target in df.columns:
            return df[[target]]
        # Fallback scan
        for c in df.columns:
            if "{key.lower()}" in c:
                return df[[c]]
        return None
"""
        registry.register(FeatureMetadata(
            feature_id=f"MACRO_{key}",
            name=f"Macro {key}",
            category="MACRO",
            description=f"Macro Economic Data: {key}",
            params=[],
            code_snippet=snippet,
            handler_func=f"Macro{key}Handler"
        ), overwrite=True)
        print(f"Registered MACRO_{key}")

    # 2. Time / Seasonality
    snippet_time = """
class ContextTimeHandler:
    def compute(self, df, **kwargs):
        from src.features.context_engineering import add_time_features
        res = add_time_features(df)
        cols = [c for c in res.columns if 'time_' in c]
        return res[cols]
"""
    registry.register(FeatureMetadata(
        feature_id="CONTEXT_TIME",
        name="Cyclical Time",
        category="CONTEXT",
        description="Cyclical Time Features (Seasonality)",
        params=[],
        code_snippet=snippet_time,
        handler_func="ContextTimeHandler"
    ), overwrite=True)
    print("Registered CONTEXT_TIME")

    # 3. Statistical Moments (Fat Tail)
    snippet_stat = """
class ContextStatHandler:
    def compute(self, df, **kwargs):
        from src.features.context_engineering import add_statistical_features
        window = kwargs.get('window', 20)
        res = add_statistical_features(df, 'close', windows=[window])
        cols = [c for c in res.columns if 'stat_' in c]
        return res[cols]
"""
    registry.register(FeatureMetadata(
        feature_id="CONTEXT_STAT",
        name="Statistical Moments",
        category="CONTEXT",
        description="Statistical Moments (Skew/Kurtosis)",
        params=[
            FeatureParam("window", "int", 10, 60, 10, 20)
        ],
        code_snippet=snippet_stat,
        handler_func="ContextStatHandler"
    ), overwrite=True)
    print("Registered CONTEXT_STAT")
    
    # 4. Relative Value (SPY Ratio)
    snippet_rel = """
class ContextRelSpyHandler:
    def compute(self, df, **kwargs):
        from src.features.context_engineering import add_relative_features
        # Look for SPY
        ctx = [c for c in df.columns if 'spy' in c]
        if not ctx: return None
        res = add_relative_features(df, 'close', context_cols=ctx)
        cols = [c for c in res.columns if 'rel_' in c]
        return res[cols]
"""
    registry.register(FeatureMetadata(
        feature_id="REL_SPY",
        name="Relative Strength SPY",
        category="CONTEXT",
        description="Relative Strength vs SPY",
        params=[],
        code_snippet=snippet_rel,
        handler_func="ContextRelSpyHandler"
    ), overwrite=True)
    print("Registered REL_SPY")

if __name__ == "__main__":
    register_context_features()
