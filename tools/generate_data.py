
import logging
import sys
import os
from pathlib import Path

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.shared.logger import get_logger
from src.features.registry import FeatureRegistry
from src.ledger.repo import LedgerRepo
from src.l3_meta.agent import MetaAgent
from src.l3_meta.detectors.regime import RegimeDetector
from src.orchestration.evaluation import evaluate_stage, persist_best_samples
from src.data.loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_data")

def generate_dashboard_data():
    logger.info(">>> [Tool] Starting Data Generation for Dashboard...")
    
    # 1. Setup
    registry = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH))
    registry.initialize()
    
    repo = LedgerRepo(config.LEDGER_DIR)
    agent = MetaAgent(registry, repo)
    detector = RegimeDetector()
    
    # 2. Load Data
    logger.info(f">>> [Data] Fetching data for {config.TARGET_TICKER}...")
    loader = DataLoader(target_ticker=config.TARGET_TICKER, start_date=config.DATA_START_DATE)
    df = loader.fetch_all()
    
    if df.empty:
        logger.error("Failed to load data.")
        return

    # 3. Detect Regime
    regime = detector.detect(df)
    logger.info(f">>> [Regime] {regime.label}")
    
    # 4. Generate a small batch
    N_SAMPLES = 5
    logger.info(f">>> [Batch] Generatng {N_SAMPLES} experiments...")
    
    history = repo.load_records()
    policies = []
    
    # Force diverse policies
    for _ in range(N_SAMPLES):
        pol = agent.propose_policy(regime, history)
        policies.append(pol)
        
    # 5. Evaluate
    logger.info(">>> [Eval] Running evaluation (this may take a moment)...")
    # Using 'full' stage to ensure compliance with main flow, but with small n_jobs
    results = evaluate_stage(policies, df, "full", regime.label, n_jobs=2)
    
    # 6. Save
    saved = persist_best_samples(repo, results, df)
    logger.info(f">>> [Save] Saved {saved} records to ledger.")
    
    # 7. Update Agent
    for r in results:
        # Simple reward mockup
        agent.learn(r.score, regime, r.policy_spec)
        
    logger.info(">>> [Complete] Data generation finished.")

if __name__ == "__main__":
    generate_dashboard_data()
