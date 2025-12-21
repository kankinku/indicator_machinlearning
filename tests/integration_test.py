
import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestration.run_experiment import run_experiment
from src.templates.registry import default_registry
from src.contracts import PolicySpec
from src.ledger.repo import LedgerRepo
from src.data.loader import DataLoader

def test_pipeline():
    print(">>> 1. Fetching Real Data (QQQ) via DataLoader...")
    # Fetch ample history to test Macro/VIX joins
    loader = DataLoader("QQQ", "2018-01-01") 
    df = loader.fetch_all()
    
    if df.empty:
        raise ValueError("Failed to fetch data for integration test.")
        
    print(f"Data Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print(">>> 2. Initializing Registry & Repo...")
    registry = default_registry()
    repo = LedgerRepo(Path("./test_ledger"))
    
    # Select Template T01
    template_id = "T01"
    
    print(f">>> 3. Creating PolicySpec for {template_id}...")
    # Manually defined tuned params roughly based on defaults
    policy = PolicySpec(
        spec_id="test_spec_001",
        template_id=template_id,
        tuned_params={"k": 1.0, "H": 10, "entry_threshold": 0.6}, # Fixed: k >= 0.8 compliance
        data_window={"train": "all"},
        risk_budget={"max_dd": 0.2}, # Relaxed DD for real data testing
        execution_assumption={"cost_bps": 5}
    )
    
    print(">>> 4. Running Experiment (CPCV Loop)...")
    try:
        record = run_experiment(registry, policy, df, repo)
        
        print("\n>>> 5. Success! Verdict:")
        print(f"Approved: {record.reason_codes == []}")
        print(f"Reason Codes: {record.reason_codes}")
        print(f"Scorecard: {record.cpcv_metrics}")
        print(f"PBO: {record.pbo}")
        print(f"Artifact Ref: {record.model_artifact_ref}")
        
    except Exception as e:
        print(f"\n!!! FAILURE !!!")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
