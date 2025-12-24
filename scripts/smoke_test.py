
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.orchestration.infinite_loop import infinite_loop

def run_smoke_test():
    print(">>> Starting Smoke Test (Step 1 Instrumentation)")
    
    # 1. Setup minimal test config
    os.environ["ENV"] = "test"
    os.environ["MAX_EXPERIMENTS"] = "5" # Very small
    os.environ["PARALLEL_BATCH_SIZE"] = "2"
    os.environ["SLEEP_INTERVAL"] = "1"
    
    # Override config for fast run
    config.MAX_EXPERIMENTS = 4 # Total experiments across batches
    config.PARALLEL_BATCH_SIZE = 2
    
    report_file = config.LOG_DIR / "instrumentation" / "batch_reports.jsonl"
    if report_file.exists():
        os.remove(report_file)
        print(f"Cleaned old report: {report_file}")

    try:
        # Run the loop. It should exit after 2-3 batches due to MAX_EXPERIMENTS
        # Wait, infinite_loop doesn't exit automatically on MAX_EXPERIMENTS unless counter > max_exps
        # Let's wrap it in a thread or just run it and hope it prunes and exits?
        # Actually infinite_loop is 'while True'. 
        # I'll modify infinite_loop to accept a 'once' or 'n_batches' parameter for testing if needed.
        # For now, let's just run it and see if the first batch logs.
        
        print(">>> Running infinite_loop for 2 iterations...")
        # Since I can't easily stop it without signal or once flag, 
        # I'll create a mini-loop version here or modify infinite_loop.
        
        # Modified infinite_loop to allow exit condition? 
        # No, let's just simulate 2 batches manually here.
        
        from src.orchestration.infinite_loop import get_registry, LedgerRepo, MetaAgent, RegimeDetector, get_epsilon_manager, DataLoader, get_curriculum_controller, get_instrumentation, evaluate_v12_batch
        from src.shared.logger import setup_main_logging, stop_main_logging
        
        # Setup Logging
        setup_main_logging()
        
        # Manual execution of 2 batches
        registry = get_registry()
        registry.warmup()
        repo = LedgerRepo(config.LEDGER_DIR)
        agent = MetaAgent(registry, repo)
        detector = RegimeDetector()
        inst = get_instrumentation()
        
        loader = DataLoader(target_ticker="QQQ", start_date="2024-01-01")
        df = loader.fetch_all()
        df["regime_label"] = detector.detect_series(df)
        agent.set_market_data(df)
        
        for b_idx in range(1, 3):
            print(f">>> Executing Test Batch {b_idx}")
            batch_seed = 2025 + b_idx
            inst.start_batch(b_idx, batch_seed, 2)
            
            regime = detector.detect(df)
            history = repo.load_records()
            
            policies = agent.propose_batch(regime, history, 2, seed=batch_seed)
            results, _ = evaluate_v12_batch(policies, df, regime.label, n_jobs=2)
            
            # Simplified quality recording
            inst.record_quality(max([r.score for r in results]), 0, 0)
            inst.end_batch()
            
        if report_file.exists():
            with open(report_file, "r") as f:
                lines = f.readlines()
                print(f"Report check: Found {len(lines)} batches in {report_file}")
                for line in lines:
                    print(f"  {line.strip()}")
            if len(lines) >= 2:
                print(">>> Smoke Test PASSED!")
            else:
                print(">>> Smoke Test FAILED: Report incomplete.")
        else:
            print(">>> Smoke Test FAILED: Report file not found.")

    except Exception as e:
        print(f"!!! Smoke Test ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        from src.shared.logger import stop_main_logging
        stop_main_logging()

if __name__ == "__main__":
    run_smoke_test()
