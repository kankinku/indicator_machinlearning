
from __future__ import annotations

import time
import logging
import multiprocessing
import os
from typing import Optional

# Prevent joblib WinError 2 by setting explicit CPU count
os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())

from src.config import config
from src.shared.logger import get_logger
from src.features.registry import FeatureRegistry
from src.ledger.repo import LedgerRepo
from src.l3_meta.agent import MetaAgent
from src.l3_meta.detectors.regime import RegimeDetector
from src.orchestration.evaluation import evaluate_stage, persist_best_samples
from src.data.loader import DataLoader

logger = get_logger("orchestration.loop")

def infinite_loop(
    target_ticker: Optional[str] = None,
    ledger_path: Optional[str] = None,
    max_experiments: Optional[int] = None,
    sleep_interval: Optional[int] = None
):
    """
    The Main Vibe Loop with Live Data.
    """
    # 0. Config Overrides
    ticker = target_ticker or config.TARGET_TICKER
    l_path = ledger_path or config.LEDGER_DIR
    max_exps = max_experiments if max_experiments is not None else config.MAX_EXPERIMENTS
    sleep_sec = sleep_interval if sleep_interval is not None else config.SLEEP_INTERVAL

    # 1. Setup
    logger.info(">>> [System] Initializing Vibe Engine (Dynamic Evolved)...")
    
    # Initialize Dynamic Registry
    registry = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH))
    registry.initialize()
    
    repo = LedgerRepo(l_path)
    agent = MetaAgent(registry, repo)
    detector = RegimeDetector()
    
    # [Config] Suppress Spam Logs
    logging.getLogger("meta.agent").setLevel(logging.INFO)
    logging.getLogger("meta.q_learning").setLevel(logging.INFO)
    logging.getLogger("feature.registry").setLevel(logging.WARNING)
    logging.getLogger("feature.custom_loader").setLevel(logging.WARNING)
    logging.getLogger("data.loader").setLevel(logging.WARNING)
    logging.getLogger("l2.ml_guard").setLevel(logging.WARNING)
    
    
    # 2. Data Load (Live Fetch)
    try:
        logger.info(f">>> [Data] Initializing DataLoader for {ticker}...")
        loader = DataLoader(target_ticker=ticker, start_date=config.DATA_START_DATE)
        df = loader.fetch_all()
        
        if df.empty:
            raise ValueError("Fetched data is empty.")
            
        logger.info(f">>> [Data] Loaded {len(df)} bars (QQQ + SPY + VIX + Macro)")
    except Exception as e:
        logger.error(f"!!! [Error] Failed to load data: {e}", exc_info=True)
        return

    # Resume Counter from History
    try:
        existing_records = repo.load_records()
        counter = len(existing_records)
        logger.info(f">>> [System] Found {counter} past experiments. Resuming from #{counter + 1}...")
    except Exception as e:
        counter = 0
        logger.warning(f">>> [System] Starting fresh (History load failed: {e}).")

    batch_idx = 0
    # 3. The Loop
    while True:
        # Performance Based Pruning (Maintenance)
        if max_exps > 0 and counter > max_exps:
            pruned = repo.prune_experiments(keep_n=max_exps)
            if pruned > 0:
                logger.info(f">>> [System] Pruned {pruned} poor performers to maintain pool size {max_exps}.")

        batch_idx += 1

        # A. Detect Regime (Situation Awareness)
        regime = detector.detect(df)

        # B. Load History (Memory)
        history = repo.load_records()

        # C. Agent Proposes Policy BATCH (Intelligence)
        # ---------------------------------------------
        n_jobs = multiprocessing.cpu_count()

        logger.info("-" * 90)
        # C. Sequential Execution (Feedback Loop)
        # ---------------------------------------------
        # Execute experiments sequentially to provide immediate feedback to the RL agent.
        # This avoids process spawning overhead and allows faster evolution.
        
        n_steps = multiprocessing.cpu_count()
        logger.info("-" * 90)
        logger.info(f"  >>> [Sequential] Starting {n_steps} Experiments (Regime: {regime.label})...")

        for i in range(n_steps):
            # 1. Propose (Intelligence)
            pol = agent.propose_policy(regime, history)
            
            # 2. Evaluate (Execution)
            # Pass n_jobs=1 to force sequential processing in the evaluator
            try:
                results = evaluate_stage([pol], df, "full", regime.label, n_jobs=1)
                
                # 3. Persist
                saved = persist_best_samples(repo, results, df)
                if saved > 0:
                    counter += saved
                
                # 4. Learn (Evolution)
                # We expect exactly one result since we sent one policy
                if results:
                    res = results[0]
                    reward = res.score
                    
                    status_icon = "OK" if reward > config.EVAL_SCORE_MIN else "NO"
                    current_exp_id = counter 
                    
                    logger.info(
                        f"  [{current_exp_id}] {status_icon} {pol.template_id:<20} | Score: {reward:>6.3f}"
                    )
                    
                    # Immediate Feedback
                    agent.learn(reward, regime, pol)
                    
                    # Optional: Update in-memory history if strictly needed, 
                    # but typically reloading from disk next batch is sufficient for deduplication.
                    # The Q-Table is already updated in memory.

            except Exception as e:
                logger.error(f"!!! [Sequential Error] {e}", exc_info=True)

        logger.info("  >>> [Sequential] Batch Complete.")



        # F. Iterate
        time.sleep(sleep_sec)

if __name__ == "__main__":
    infinite_loop()
