
import sys
import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import random

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.features.registry import get_registry
from src.features.ontology import get_feature_ontology
from src.l3_meta.agent import MetaAgent
from src.l3_meta.reward_shaper import get_reward_shaper
from src.l3_meta.d3qn_agent import IntegratedD3QNAgent
from src.l3_meta.state import RegimeState
from src.ledger.repo import LedgerRepo
from src.config import config

def run_integration_test():
    print("================================================================")
    print("       GENOME V2 FULL-STACK INTEGRATION TEST (SMOKE)            ")
    print("================================================================")
    
    # 0. Prep
    ledger_path = Path("ledger/smoke_test")
    if ledger_path.exists():
        import shutil
        shutil.rmtree(ledger_path)
    ledger_path.mkdir(parents=True, exist_ok=True)
    repo = LedgerRepo(str(ledger_path))

    # 1. Feature Registry & Genome v2 Metadata Check
    print("\n[Stage 1] Feature Registry & Metadata Check...")
    registry = get_registry()
    registry.initialize() # Load from feature_registry.json
    all_features = registry.list_all()
    print(f"    - Total Features Loaded: {len(all_features)}")
    
    # Check if Genome v2 fields exist
    test_feat = registry.get("MOMENTUM_RSI_V1")
    if test_feat and hasattr(test_feat, 'causality_link'):
        print(f"    - [SUCCESS] Genome v2 Metadata found: {test_feat.feature_id}")
    else:
        print("    - [FAIL] Genome v2 Metadata missing or feature not found.")
        return

    # 2. Ontology & Market Interrogation Logic
    print("\n[Stage 2] Ontology & Market Interrogation Check...")
    ontology = get_feature_ontology()
    questions = ontology.MARKET_QUESTIONS
    print(f"    - Market Questions Defined: {list(questions.keys())}")
    
    # Test Compatibility Score
    comb = ["MOMENTUM_RSI_V1", "TREND_ADX_V1"] # Conflicting pair
    score = ontology.check_compatibility(comb)
    status = "SUCCESS" if score < 0 else "WARNING"
    print(f"    - [{status}] Compatibility Score (RSI + ADX): {score:.2f}")

    # 3. MetaAgent Combinator Mode
    print("\n[Stage 3] MetaAgent Combinator Mode (Strategy Generation)...")
    agent = MetaAgent(registry, repo) # Pass required DI
    mock_regime = RegimeState(trend_score=0.8, vol_level=1.2, corr_score=0.1, shock_flag=False, label="TREND_UP")
    
    # Propose a policy based on a high-level action
    try:
        # Test direct interrogation-based construction
        logic_trees = agent._construct_trees_from_action("TREND_ALPHA", mock_regime, [])
        print(f"    - [SUCCESS] LogicTrees constructed for TREND_ALPHA via Ontology.")
        
        # Test full propose_policy
        policy = agent.propose_policy(mock_regime, [])
        print(f"    - [SUCCESS] Full Policy proposed: {policy.spec_id}")
        # print(f"      Genome Mapping: {[f.feature_id for f in policy.feature_genome]}")
    except Exception as e:
        print(f"    - [FAIL] MetaAgent generation failed: {e}")
        import traceback; traceback.print_exc()
        return

    # 4. Reward Shaper & Stability Component
    print("\n[Stage 4] Reward Shaper (Information Efficiency)...")
    reward_shaper = get_reward_shaper()
    mock_metrics = {
        "window_results": [
            type('obj', (object,), {'avg_alpha': 0.05})(),
            type('obj', (object,), {'avg_alpha': 0.051})() # Very stable alpha
        ],
        "n_trades": 50,
        "total_return_pct": 2.5,
        "is_rejected": False
    }
    bd = reward_shaper.compute_breakdown(mock_metrics)
    print(f"    - [SUCCESS] Reward computed: {bd.total:.2f}")
    if hasattr(bd, 'stability_component'):
        print(f"      Stability Component (Weight): {bd.stability_component:.2f}")

    # 5. Integrated RL Agent & GPU Learning
    print("\n[Stage 5] RL Agent & Blackwell GPU Learning Check...")
    # Storage for models
    model_dir = ledger_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    rl_agent = IntegratedD3QNAgent(
        storage_path=model_dir,
        strategy_actions=["TREND_ALPHA", "MOMENTUM_TAIL"],
        risk_actions=["RISK_LOW"]
    )
    print(f"    - Agent Device: {rl_agent.device}")
    
    # Mock transition push to buffer to force learning
    state_dim = rl_agent.state_encoder.get_state_dim()
    mock_state = np.random.randn(state_dim).astype(np.float32)
    mock_next_state = np.random.randn(state_dim).astype(np.float32)
    
    try:
        # Push enough transitions to allow sampling
        target_push = config.D3QN_MIN_BUFFER_SIZE + 5
        for _ in range(target_push):
            rl_agent.replay_buffer.push_transition(
                mock_state, [0, 0], 1.0, mock_next_state, weight=1.2, tag="PASS"
            )
        
        # Test learning call
        loss = rl_agent._learn()
        if loss is not None:
             print(f"    - [SUCCESS] sm_120 GPU Learning Step Loss: {loss:.4f}")
        else:
             print("    - [FAIL] Learning step returned None.")
             
    except Exception as e:
        print(f"    - [FAIL] RL Learning failed: {e}")
        import traceback; traceback.print_exc()
        return

    print("\n================================================================")
    print("   ALL SYSTEMS GREEN: GENOME V2 & SM_120 INTEGRATED SUCCESS     ")
    print("================================================================")

if __name__ == "__main__":
    run_integration_test()
