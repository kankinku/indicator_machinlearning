
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from src.l3_meta.agent import MetaAgent
from src.ledger.repo import LedgerRepo
from src.l3_meta.state import RegimeState
import pandas as pd
import numpy as np

def test_integrated_agent():
    repo = LedgerRepo(PROJECT_ROOT / "ledger")
    agent = MetaAgent(registry=None, repo=repo)
    
    # Mock Regime
    regime = RegimeState(
        label="BULL",
        trend_score=1.5,
        vol_level=0.2,
        corr_score=0.5,
        shock_flag=False
    )
    
    # Mock DF
    df = pd.DataFrame({
        "close": np.random.randn(100),
        "volume": np.random.randn(100)
    })
    
    print("Testing Integrated Agent Action Selection...")
    # Get Action
    policy = agent.propose_policy(regime, history=[])
    
    print(f"Policy ID: {policy.spec_id}")
    print(f"Strategy: {policy.rl_meta.get('strategy_action')}")
    print(f"Risk: {policy.rl_meta.get('risk_profile')}")
    print(f"Features: {list(policy.feature_genome.keys())}")
    
    # Test Update
    print("\nTesting Integrated Agent Update...")
    mock_metrics = {
        "total_return_pct": 5.0,
        "mdd_pct": 2.0,
        "n_trades": 25,
        "win_rate": 0.6,
        "reward_risk": 1.5,
        "is_rejected": False
    }
    
    agent.learn(reward=1.0, next_regime=regime, policy_spec=policy, metrics=mock_metrics)
    print("Update successful.")

if __name__ == "__main__":
    test_integrated_agent()
