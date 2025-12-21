
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import time

from src.config import config
from src.features.registry import FeatureRegistry
from src.features.factory import FeatureFactory
from src.l3_meta.agent import MetaAgent
from src.l3_meta.detectors.regime import RegimeDetector
from src.l2_sl.model import TacticalModel
from src.l1_judge.manager import OprManagerRL, PositionState
from src.ledger.repo import LedgerRepo

def test_full_ecosystem():
    print("\n=== 🪐 Starting Full Ecosystem Simulation (L3 -> L2 -> L1) ===\n")
    
    # 0. Setup Environment
    # Clean up artifacts for fresh test
    test_artifact_dir = Path("./tests/artifacts_sim")
    if test_artifact_dir.exists():
        shutil.rmtree(test_artifact_dir)
    test_artifact_dir.mkdir(parents=True)
    
    # 1. Data Generation (Mock Market Data)
    print("[1] Generating Mock Reality (Market Data)...")
    dates = pd.date_range(start="2024-01-01", periods=200, freq='D')
    # Create a Bull Trend then a Crash
    prices = [100.0]
    for i in range(1, 200):
        change = np.random.normal(0.005, 0.01) if i < 150 else np.random.normal(-0.02, 0.02)
        prices.append(prices[-1] * (1 + change))
        
    df = pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000000] * 200,
        "spy_close": prices, # correlated
        "vix_close": [15] * 150 + [35] * 50, # Spike in crash
        "tlt_close": [100] * 200,
    }, index=dates)
    print(f"    - Data Shape: {df.shape}")
    
    # 2. L3 Meta-Agent (The Director)
    print("\n[2] L3 Director Awakening...")
    registry = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH))
    registry.initialize()
    
    regime_detector = RegimeDetector()
    current_regime = regime_detector.detect(df.iloc[:100]) # Early Bull Phase
    print(f"    - Detected Regime: {current_regime.label} (Trend Score: {current_regime.trend_score:.2f})")
    
    # Mock Repo
    class MockRepo:
        def __init__(self): self.base_dir = test_artifact_dir
        
    l3_agent = MetaAgent(registry, MockRepo())
    policy = l3_agent.propose_policy(current_regime, [])
    print(f"    - L3 Proposal: {policy.template_id}")
    print(f"    - Genes Selected: {list(policy.feature_genome.keys())}")
    
    # 3. L2 Tactical Engine (The Hunter)
    print("\n[3] L2 Hunter Training...")
    factory = FeatureFactory()
    features = factory.generate_from_genome(df, policy.feature_genome)
    
    l2_model = TacticalModel(model_type="rf")
    # Train on first 100 days
    l2_model.train(features.iloc[:100], df["close"].iloc[:100])
    print("    - L2 Model Trained.")
    
    # Predict on day 101 (Simulate Live)
    today_feat = features.iloc[[100]]
    confidence = l2_model.predict_uncertainty(today_feat).iloc[0]
    print(f"    - Hunter Signal (Day 101): Confidence {confidence:.2f}")
    
    # 4. L1 Operational Manager (The Manager)
    print("\n[4] L1 Manager Execution...")
    l1_manager = OprManagerRL(test_artifact_dir)
    
    # Simulate a position lifecycle
    entry_price = df["close"].iloc[100]
    position_state = PositionState(
        entry_price=entry_price,
        current_price=entry_price * 1.02, # 2% Profit
        size=1.0,
        duration=5,
        confidence=confidence,
        atr=1.5,
        stop_loss=entry_price * 0.98
    )
    
    action, _ = l1_manager.decide_action(position_state)
    print(f"    - State: Profit +2%, Duration 5d, Conf {confidence:.2f}")
    print(f"    - Manager Decision: {action}")
    
    # Check if Q-Table updates
    next_state = PositionState(
        entry_price=entry_price,
        current_price=entry_price * 1.05, # 5% Profit
        size=0.5,
        duration=10,
        confidence=confidence,
        atr=1.5,
        stop_loss=entry_price * 0.99
    )
    l1_manager.learn(position_state, 0, 1.0, next_state, False) # Reward 1.0 for Holding profit
    l1_manager.save()
    print("    - Manager Learned & Saved.")
    
    print("\n=== 🪐 Ecosystem Simulation Complete: All Systems GO ===")

if __name__ == "__main__":
    test_full_ecosystem()
