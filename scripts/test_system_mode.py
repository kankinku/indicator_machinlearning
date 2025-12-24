from __future__ import annotations
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

from src.testing.scenario import Scenario, DataScenario, PolicyScenario
from src.testing.runner import TestRunner
from src.l3_meta.state import RegimeState
from src.l3_meta.agent import MetaAgent
from src.features.registry import get_registry
from src.ledger.repo import LedgerRepo
from pathlib import Path
import pandas as pd
import numpy as np

def demo_workflow():
    """A simplified version of the infinite loop for testing."""
    registry = get_registry()
    repo = LedgerRepo(Path("test_ledger")) 
    agent = MetaAgent(registry, repo)
    
    # Mock data
    dates = pd.date_range("2023-01-01", periods=100)
    df = pd.DataFrame({
        "open": np.random.randn(100) + 100,
        "high": np.random.randn(100) + 101,
        "low": np.random.randn(100) + 99,
        "close": np.random.randn(100) + 100,
        "volume": np.random.randn(100) * 1000 + 5000
    }, index=dates)
    
    regime = RegimeState(trend_score=0.8, vol_level=0.2, corr_score=0.5, shock_flag=False, label="BULL")
    history = []
    
    # Run 1 batch (sequential mode for simplicity in test)
    from src.orchestration.infinite_loop import _run_batch_sequential
    _run_batch_sequential(agent, df, regime, history, repo, batch_size=2)

if __name__ == "__main__":
    runner = TestRunner()
    
    # Test 1: Standard Pass
    sc1 = Scenario(
        name="STANDARD_RUN",
        data=DataScenario(name="UPTREND", behavior="PERFECT_UPTREND")
    )
    # Note: Standard runner.validate might need more logic, but for demo:
    runner.run_case("STANDARD_RUN_DEMO", sc1, demo_workflow)
    
    # Test 2: Drop everything in Stage 1
    sc2 = Scenario(
        name="STAGE1_DROP",
        data=DataScenario(name="NO_TRADE", behavior="NO_TRADE")
    )
    runner.run_case("STAGE1_DROP", sc2, demo_workflow)

    # Test 3: Forced Rigid (AutoTuner check)
    from src.testing.scenario import EnvironmentScenario
    sc3 = Scenario(
        name="FORCE_RIGID_TEST",
        env=EnvironmentScenario(name="RIGID", force_rigid=True)
    )
    # We need to call process_diagnostics to trigger tuner
    def workflow_with_tuner():
        demo_workflow()
        from src.l3_meta.auto_tuner import get_auto_tuner
        tuner = get_auto_tuner()
        # Mock diagnostic and extra info
        # We need at least 3 history entries to trigger analysis
        for i in range(3):
            diag = {"pass_rate": 0.05}
            extra = {"batch_id": 10 + i, "pass_rate_s1": 0.05, "regime": "BULL"}
            tuner.process_diagnostics(diag, extra)

    runner.run_case("FORCE_RIGID_TEST", sc3, workflow_with_tuner)
