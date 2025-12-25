
import pytest
from unittest.mock import MagicMock
from src.contracts import ColumnRef, FeatureMetadata
from src.shared.column_resolver import ColumnResolver, ColumnResolutionError
from src.features.registry import inject_registry, reset_registry
from src.config import config

# Mock Registry
class MockRegistry:
    def get(self, feature_id):
        if feature_id == "RSI_V1":
            return FeatureMetadata(
                feature_id="RSI_V1", 
                name="RSI", 
                category="Momentum",
                description="", code_snippet="", handler_func="", params=[],
                outputs={"value": "rsi"}  # Explicit mapping
            )
        if feature_id == "SMA_V1":
            return FeatureMetadata(
                feature_id="SMA_V1", 
                name="SMA", 
                category="Trend",
                description="", code_snippet="", handler_func="", params=[],
                outputs={"value": "value"} # Default mapping
            )
        if feature_id == "MISSING_COL_FEAT":
            return FeatureMetadata(
                 feature_id="MISSING_COL_FEAT",
                 name="Missing", category="Test", description="", code_snippet="", handler_func="", params=[],
                 outputs={"value": "val"}
            )
        return None

@pytest.fixture
def setup_env():
    # Setup
    reset_registry()
    inject_registry(MockRegistry())
    config.LOGICTREE_STRICT = True
    yield
    # Teardown
    reset_registry()

def test_resolve_explicit(setup_env):
    df_cols = ["RSI_V1__rsi", "SMA_V1__value", "MACD_V1__signal"]
    resolver = ColumnResolver(df_cols)
    
    ref = ColumnRef("RSI_V1", "value")
    col = resolver.resolve(ref)
    assert col == "RSI_V1__rsi"

def test_resolve_default(setup_env):
    df_cols = ["RSI_V1__rsi", "SMA_V1__value"]
    resolver = ColumnResolver(df_cols)
    
    ref = ColumnRef("SMA_V1", "value")
    col = resolver.resolve(ref)
    assert col == "SMA_V1__value"

def test_resolve_missing_feature(setup_env):
    resolver = ColumnResolver([])
    ref = ColumnRef("UNKNOWN_FEAT", "value")
    
    with pytest.raises(ColumnResolutionError) as excinfo:
        resolver.resolve(ref)
    assert excinfo.value.reason == "FEATURE_NOT_REGISTERED"

def test_resolve_missing_output_key(setup_env):
    resolver = ColumnResolver([])
    ref = ColumnRef("RSI_V1", "signal") # RSI only has value in mock
    
    with pytest.raises(ColumnResolutionError) as excinfo:
        resolver.resolve(ref)
    assert excinfo.value.reason == "OUTPUT_KEY_UNDEFINED"

def test_resolve_column_not_in_df(setup_env):
    resolver = ColumnResolver(["OTHER__col"])
    ref = ColumnRef("RSI_V1", "value") # Should map to RSI_V1__rsi
    
    with pytest.raises(ColumnResolutionError) as excinfo:
        resolver.resolve(ref)
    assert excinfo.value.reason == "COLUMN_NOT_FOUND"
    assert "RSI_V1__rsi" in str(excinfo.value)

def test_fuzzy_fallback_in_lenient_mode(setup_env):
    config.LOGICTREE_STRICT = False
    
    # Situation: Feature registered, output key undefined, but column exists with prefix
    df_cols = ["RSI_V1__whatever"]
    resolver = ColumnResolver(df_cols)
    
    # Requesting defined feature but undefined key 'special'
    # Mock doesn't have 'special', raises OUTPUT_KEY_UNDEFINED in strict.
    # In lenient, falls back to fuzzy.
    ref = ColumnRef("RSI_V1", "special") 
    
    col = resolver.resolve(ref)
    assert col == "RSI_V1__whatever"

if __name__ == "__main__":
    pytest.main([__file__])
