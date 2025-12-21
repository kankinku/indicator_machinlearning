
import sys
import os
import pandas as pd
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import config
from src.features.registry import FeatureRegistry
from src.l3_meta.agent import MetaAgent
from src.features.factory import FeatureFactory
from src.ledger.repo import LedgerRepo
from src.l3_meta.detectors.regime import RegimeDetector

from pathlib import Path

# Mock classes for dependency injection
class MockRepo:
    def __init__(self):
        self.base_dir = Path("./mock_ledger")

def test_evolutionary_system():
    print("=== Starting System Integration Test ===")
    
    # 1. Registry Check
    print("\n[Test 1] Checking Feature Registry...")
    registry = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH))
    registry.initialize()
    features = registry.list_all()
    print(f"✅ Registry loaded {len(features)} features.")
    
    if len(features) < 21:
        print("❌ Warning: Registry count mismatch. Expected 21.")
    
    # 2. Regime Detector Check
    print("\n[Test 2] Testing Regime Detector...")
    
    # Create Dummy Data (Bullish trend)
    dates = pd.date_range(start="2023-01-01", periods=100)
    df = pd.DataFrame({
        "open": [100 + i for i in range(100)],
        "high": [105 + i for i in range(100)],
        "low": [95 + i for i in range(100)],
        "close": [102 + i*1.1 for i in range(100)], # Strong Uptrend
        "volume": [1000 for _ in range(100)]
    }, index=dates)
    
    detector = RegimeDetector()
    regime = detector.detect(df)
    print(f"✅ Detected Regime Label: {regime.label} (Score: {regime.trend_score:.2f})")
    
    if regime.label not in ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOL", "SHOCK"]:
        print(f"❌ Error: Invalid Regime Label {regime.label}")

    # 3. Agent Evolution Check
    print("\n[Test 3] Testing Evolutionary Agent (L3)...")
    repo = MockRepo()
    agent = MetaAgent(registry, repo) # Pass registry explicitly
    
    # Simulate RL Action (Agent decides based on regime)
    policy = agent.propose_policy(regime, [])
    genome = policy.feature_genome
    
    print(f"✅ Proposed Action Template: {policy.template_id}")
    print(f"✅ Evolved Genome: {list(genome.keys())}")
    
    if not genome:
        print("❌ Error: Empty genome generated.")
        return

    # 4. Dynamic Factory Execution Check
    print("\n[Test 4] Testing Dynamic Factory Execution...")
    factory = FeatureFactory()
    
    try:
        features_df = factory.generate_from_genome(df, genome)
        print(f"✅ Features generated successfully. Shape: {features_df.shape}")
        print("   Columns:", features_df.columns.tolist())
        
        if features_df.empty:
             print("❌ Error: Generated features are empty.")
    except Exception as e:
        print(f"❌ Factory Execution Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_evolutionary_system()
