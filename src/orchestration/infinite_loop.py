
from __future__ import annotations

import time
from typing import Optional

from src.config import config
from src.shared.logger import get_logger
from src.features.registry import FeatureRegistry
from src.ledger.repo import LedgerRepo
from src.l3_meta.agent import MetaAgent
from src.l3_meta.detectors.regime import RegimeDetector
from src.orchestration.run_experiment import run_experiment
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
    
    # 3. The Loop
    while True:
        if max_exps > 0 and counter >= max_exps:
            logger.info(">>> [System] Max experiments reached. Stopping.")
            break
            
        counter += 1
        logger.info(f"\n=== Experiment {counter} ===")
        
        # A. Detect Regime (Situation Awareness)
        regime = detector.detect(df)
        logger.info(f">>> [Regime] Trend={regime.trend_score:.2f}, VolLevel={regime.vol_level:.2f}, Shock={regime.shock_flag}")
        
        # B. Load History (Memory)
        history = repo.load_records() 
        
        # C. Agent Proposes Policy (Intelligence)
        policy = agent.propose_policy(regime, history)
        logger.info(f">>> [Agent] Proposed: {policy.template_id} | Params: {policy.tuned_params}")
        
        # D. Execution (Action)
        try:
            record = run_experiment(registry, policy, df, repo)
            
            # E. Report (Feedback)
            # E. Report (Feedback)
            status = "APPROVED" if not record.reason_codes else "REJECTED"
            
            # Extract Metrics
            perf = record.cpcv_metrics.get('cpcv_mean', 0.0)      # Return
            vol = record.cpcv_metrics.get('cpcv_std', 1.0)
            n_trades = record.cpcv_metrics.get('n_trades', 0)
            win_rate = record.cpcv_metrics.get('win_rate', 0.0)
            
            # Key Metric 1: Sharpe
            sharpe = perf / (vol + 1e-9) if vol > 0 else 0.0
            
            # Key Metric 2: Trade Score (Normalized to 50 trades = 1.0)
            # Cap at 1.0 to prevent high frequency spam from dominating
            trade_score = min(n_trades / 50.0, 1.0)
            
            # Key Metric 3: Win Rate (0.0 - 1.0)
            
            # Key Metric 4: Return Score (Scale 0.05 -> 0.5)
            return_score = perf * 10.0
            
            # RL Reward Shaping: Weighted Composite Score
            # Weights: Sharpe(4), Trades(1), WinRate(2), Return(2)
            if not record.reason_codes:
                reward = (4.0 * sharpe) + (1.0 * trade_score) + (2.0 * win_rate) + (2.0 * return_score)
            else:
                reward = -1.0 # Hard Penalty for Failure
            
            logger.info(f">>> [Result] {status} | Reward: {reward:.3f} (S:{sharpe:.2f} T:{n_trades} W:{win_rate:.2f} R:{perf:.3f})")
            
            # Update Agent (RL Feedback)
            agent.learn(reward, regime)

            if record.fix_suggestion:
                logger.info(f"    [Fix] Suggestion: {record.fix_suggestion}")
                
        except Exception as e:
            logger.error(f"!!! [Experiment Error] {e}", exc_info=True)

        # F. Iterate
        time.sleep(sleep_sec)

if __name__ == "__main__":
    infinite_loop()
