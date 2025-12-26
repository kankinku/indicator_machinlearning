"""
Infinite Loop - 자율 진화 실험 엔진

이 모듈은 메인 실험 루프를 담당합니다.

성능 최적화:
1. 병렬 처리: joblib Parallel을 사용한 다중 실험 동시 실행
2. 배치 학습: RL 에이전트는 배치 완료 후 일괄 학습 (Race Condition 방지)
3. 메모리 캐시: DataFrame 해시 기반 빠른 캐싱
"""
from __future__ import annotations

import time
import logging
import multiprocessing
import os
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

# Prevent joblib WinError 2 by setting explicit CPU count
os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())

# Prevent joblib WinError 2 by setting explicit CPU count
os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())

from joblib import Parallel, delayed

from src.config import config
from src.shared.logger import get_logger
from src.shared.caching import get_feature_cache, clear_all_caches
from src.features.registry import get_registry, inject_registry
from src.ledger.repo import LedgerRepo
from src.l3_meta.agent import MetaAgent
from src.l3_meta.detectors.regime import RegimeDetector
from src.l3_meta.curriculum_controller import get_curriculum_controller
from src.l3_meta.epsilon_manager import get_epsilon_manager
from src.l3_meta.reward_shaper import get_reward_shaper
from src.orchestration.evaluation import evaluate_stage, evaluate_v12_batch, persist_best_samples, ModuleResult
from src.data.loader import DataLoader
from src.contracts import PolicySpec
from src.shared.instrumentation import get_instrumentation

logger = get_logger("orchestration.loop")


@dataclass
class ExperimentResult:
    """단일 실험 결과를 담는 데이터 클래스."""
    policy: PolicySpec
    score: float
    saved: int
    success: bool
    error: Optional[str] = None
    metrics: Optional[dict] = None  # D3QN 보상 계산용 상세 지표


# [V16] _extract_metrics removed. Using unified metrics from EvaluationResult.


def _run_single_experiment(
    policy: PolicySpec,
    df,
    regime_label: str,
    preloaded_registry=None,  # [Optim] 메인 프로세스에서 전달받은 레지스트리
) -> ExperimentResult:
    """
    단일 실험을 실행합니다. (병렬 처리용 워커 함수)
    
    Note:
        이 함수는 별도 프로세스에서 실행될 수 있으므로 상태를 공유하지 않습니다.
        RL 학습은 여기서 수행하지 않습니다 (메인 프로세스에서 일괄 수행).
    """
    try:
        # [Optim] 워커 프로세스 초기화: 레지스트리 주입
        if preloaded_registry is not None:
            inject_registry(preloaded_registry)

        # 평가 실행 (n_jobs=1로 내부 병렬화 비활성화 - 이미 외부에서 병렬화됨)
        results, status = evaluate_stage([policy], df, "full", regime_label, n_jobs=1)
        
        if results:
            res = results[0]
            
            # 메트릭 추출 (D3QN용)
            metrics_dict = None
            if res.best_sample and res.best_sample.metrics:
                m = res.best_sample.metrics
                # [V9] Backtest 기반 필드명
                metrics_dict = {
                    "total_return_pct": m.equity.total_return_pct,
                    "mdd_pct": m.equity.max_drawdown_pct,
                    "win_rate": m.trades.win_rate,
                    "n_trades": m.trades.trade_count,
                    "valid_trade_count": m.trades.valid_trade_count if hasattr(m.trades, "valid_trade_count") else m.trades.trade_count,
                    "benchmark_roi_pct": m.equity.benchmark_roi_pct if hasattr(m.equity, "benchmark_roi_pct") else 0.0,
                    "sharpe": m.equity.sharpe if hasattr(m.equity, 'sharpe') else 0.0,
                    "cpcv_mean": m.raw_score, # Placeholder or map from raw_score
                    "cpcv_std": 0.0,
                    "trades_per_year": m.trades.trades_per_year if hasattr(m.trades, 'trades_per_year') else 0.0,
                    "profit_factor": m.trades.profit_factor if hasattr(m.trades, 'profit_factor') else 1.0,
                }
            
            return ExperimentResult(
                policy=policy,
                score=res.score,
                saved=0,  # 저장은 메인 프로세스에서 수행
                success=True,
                metrics=metrics_dict,
            )
        else:
            return ExperimentResult(
                policy=policy,
                score=config.EVAL_SCORE_MIN,
                saved=0,
                success=False,
                error="No evaluation results",
            )
    except Exception as e:
        return ExperimentResult(
            policy=policy,
            score=config.EVAL_SCORE_MIN,
            saved=0,
            success=False,
            error=str(e),
        )


def _run_batch_parallel(
    policies: List[PolicySpec],
    df,
    regime_label: str,
    n_jobs: int,
) -> List[ExperimentResult]:
    """
    여러 실험을 병렬로 실행합니다.
    
    Args:
        policies: 실행할 정책들
        df: 시장 데이터
        regime_label: 현재 시장 상태 라벨
        n_jobs: 병렬 작업 수
    
    Returns:
        실험 결과 리스트
    """
    if not policies:
        return []
    
    # [Optim] 메인 프로세스의 레지스트리를 가져옵니다 (이미 로드됨)
    main_registry = get_registry()

    # joblib Parallel 사용 - prefer="processes"로 GIL 우회
    results = Parallel(
        n_jobs=n_jobs,
        backend=config.PARALLEL_BACKEND,
        timeout=config.PARALLEL_TIMEOUT,
        verbose=0,
    )(
        delayed(_run_single_experiment)(
            policy=policy,
            df=df,
            regime_label=regime_label,
            preloaded_registry=main_registry, # [Optim] 객체 전달 (Pickling)
        )
        for policy in policies
    )
    
    return results


def _run_batch_sequential(
    agent: MetaAgent,
    df,
    regime,
    history,
    repo: LedgerRepo,
    batch_size: int,
) -> Tuple[int, List[Tuple[float, PolicySpec]]]:
    """
    순차적으로 실험을 실행하고 즉시 학습합니다.
    
    이 모드는 RL 에이전트가 즉각적인 피드백을 받을 수 있지만,
    병렬 처리의 속도 이점을 포기합니다.
    
    Returns:
        (saved_count, batch_results) - 저장된 실험 수와 결과 리스트
    """
    batch_results = []
    saved_total = 0
    
    for i in range(batch_size):
        # 1. 정책 제안
        pol = agent.propose_policy(regime, history)
        
        # 2. 평가
        try:
            results, status = evaluate_v12_batch([pol], df, regime.label, n_jobs=1)
            
            # 3. 저장
            existing_ids = {r.policy_spec.spec_id for r in history}
            saved = persist_best_samples(repo, results, df, existing_ids=existing_ids)
            saved_total += saved
            
            # 4. 즉시 학습
            if results:
                from src.shared.metrics import metrics_to_legacy_dict, aggregate_windows
                res = results[0]
                reward = res.score
                
                m_dict = {}
                if res.best_sample:
                    m_dict = metrics_to_legacy_dict(
                        aggregate_windows([res.best_sample.metrics], eval_score_override=res.score)
                    )
                    m_dict["trade_logic"] = res.best_sample.core.get("trade_logic", {})
                    m_dict["is_rejected"] = res.score <= config.EVAL_SCORE_MIN
                
                agent.learn(reward, regime, pol, metrics=m_dict)
                batch_results.append(res)
                
                status_icon = "OK" if reward > config.EVAL_SCORE_MIN else "NO"
                logger.info(
                    f"  [{i+1}] {status_icon} {pol.template_id:<20} | Score: {reward:>6.3f}"
                )
                
                # 5. [V11] Curriculum 결과 기록
                curriculum = get_curriculum_controller() if config.CURRICULUM_ENABLED else None
                if curriculum and m_dict:
                    from src.l3_meta.reward_shaper import get_reward_shaper
                    shaper = get_reward_shaper()
                    breakdown = shaper.compute_breakdown(m_dict)
                    passed, reason = curriculum.evaluate_against_current(
                        total_return_pct=m_dict.get("total_return_pct", 0.0),
                        trades_per_year=m_dict.get("trades_per_year", 0.0),
                        mdd_pct=m_dict.get("mdd_pct", 0.0),
                        win_rate=m_dict.get("win_rate", 0.0),
                        alpha=breakdown.alpha,
                        profit_factor=m_dict.get("profit_factor", 1.0),
                    )
                    status_change = curriculum.record_result(passed, m_dict)
                    if status_change.get("promoted"):
                        logger.info(f"  >>> [Curriculum] PROMOTED to Stage {status_change['stage_after']}!")
        except Exception as e:
            exc_type = type(e).__name__
            from src.shared.instrumentation import get_instrumentation
            get_instrumentation().record_exception(exc_type)
            logger.error(f"!!! [Sequential Error] {e}", exc_info=True)
    
    return saved_total, batch_results


def infinite_loop(
    target_ticker: Optional[str] = None,
    ledger_path: Optional[str] = None,
    max_experiments: Optional[int] = None,
    sleep_interval: Optional[int] = None
):
    """
    메인 자율 진화 루프.
    
    두 가지 실행 모드를 지원합니다:
    1. 병렬 모드 (PARALLEL_ENABLED=True): 빠른 탐색, 배치 후 일괄 학습
    2. 순차 모드 (PARALLEL_ENABLED=False): 느리지만 즉각적인 피드백
    """
    # 0. Config Overrides
    ticker = target_ticker or config.TARGET_TICKER
    l_path = ledger_path or config.LEDGER_DIR
    max_exps = max_experiments if max_experiments is not None else config.MAX_EXPERIMENTS
    sleep_sec = sleep_interval if sleep_interval is not None else config.SLEEP_INTERVAL

    # 1. Setup
    logger.info(">>> [System] Initializing Vibe Engine (Dynamic Evolved)...")
    
    # DI: 싱글톤 레지스트리 사용 (중복 초기화 방지)
    registry = get_registry()
    registry.warmup()  # [Optim] 핸들러 사전 컴파일 (메인 프로세스 1회)
    
    repo = LedgerRepo(l_path)
    agent = MetaAgent(registry, repo)
    detector = RegimeDetector()
    eps_manager = get_epsilon_manager()
    instrumentation = get_instrumentation()
    
    # 병렬 설정
    n_jobs = config.PARALLEL_BATCH_SIZE or multiprocessing.cpu_count()
    parallel_mode = config.PARALLEL_ENABLED
    
    # D3QN 모드 확인
    d3qn_mode = config.D3QN_ENABLED
    logger.info(f">>> [System] Parallel Mode: {parallel_mode} | Workers: {n_jobs} | D3QN: {d3qn_mode}")
    
    # [V11] Curriculum Controller 초기화
    curriculum = get_curriculum_controller() if config.CURRICULUM_ENABLED else None
    if curriculum:
        stage_info = curriculum.get_stage_info()
        logger.info(
            f">>> [Curriculum] Stage {stage_info['current_stage']}: "
            f"{stage_info['description']} | "
            f"Passes: {stage_info['stage_passes']}/{stage_info['threshold_to_next']}"
        )
    
    # [Config] Suppress Spam Logs
    logging.getLogger("meta.agent").setLevel(logging.INFO)
    logging.getLogger("meta.q_learning").setLevel(logging.INFO)
    logging.getLogger("feature.registry").setLevel(logging.WARNING)
    logging.getLogger("feature.custom_loader").setLevel(logging.WARNING)
    logging.getLogger("data.loader").setLevel(logging.WARNING)
    logging.getLogger("l2.ml_guard").setLevel(logging.WARNING)
    logging.getLogger("l3.d3qn_agent").setLevel(logging.INFO)
    logging.getLogger("l3.state_encoder").setLevel(logging.WARNING)
    logging.getLogger("l3.curriculum").setLevel(logging.INFO)
    
    
    # 2. Data Load (Live Fetch)
    try:
        # [V8.4] Force QQQ / YFinance (Environment variable bypass for stability)
        source = config.DATA_SOURCE or "yfinance"
        target = ticker or "QQQ"
        
        logger.info(f">>> [Data] Initializing DataLoader for {target} ({source})...")
        if source == "binance":
            from src.data.binance_loader import fetch_btc_training_data
            df = fetch_btc_training_data()
        else:
            loader = DataLoader(target_ticker=target, start_date=config.DATA_START_DATE)
            df = loader.fetch_all()
            
        if df.empty:
            raise ValueError(f"Fetched data is empty for {target}")
            
        logger.info(f">>> [Data] Loaded {len(df)} bars successfully.")
        
        # [V11.2] Pre-calculate Regime Labels for backtest validation
        logger.info(">>> [System] Pre-calculating Market Regimes for validation...")
        df["regime_label"] = detector.detect_series(df)
        
        agent.set_market_data(df)
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
        batch_start_time = time.time()
        
        # Performance Based Pruning (Maintenance)
        if max_exps > 0 and counter > max_exps:
            pruned = repo.prune_experiments(keep_n=max_exps)
            if pruned > 0:
                logger.info(f">>> [System] Pruned {pruned} poor performers to maintain pool size {max_exps}.")

        batch_idx += 1
        
        # [V16] Deterministic Batch Seed
        # Formula: Base Seed (e.g. 2025) + Batch Index
        batch_seed = 2025 + batch_idx
        instrumentation.start_batch(batch_idx, batch_seed, n_jobs)

        # Defaults for batch-level reporting to avoid undefined vars
        policies = []
        results = []
        valid_results = []
        diverse_results = []
        batch_rewards = []
        diagnostic_status = {"status": "EMPTY"}
        tuner = None

        # A. Detect Regime (Situation Awareness)
        regime = detector.detect(df)

        # B. Load History (Memory)
        history = repo.load_records()
        existing_ids = {r.policy_spec.spec_id for r in history}

        # logger.info("-" * 90) removed to replace with better header
        
        if parallel_mode:
            # ========================================
            # C-1. 병렬 실행 모드
            # ========================================
            logger.info(f"  >>> [Parallel] Generating {n_jobs} Policies (Regime: {regime.label})...")
            
            # 1. 정책 배치 생성 (Diversity-aware Batch)
            policies = agent.propose_batch(regime, history, n_jobs, seed=batch_seed)
            
            # 2. 병렬 평가 (V14 Optimized)
            logger.info(f"  >>> [V14] Evaluating {len(policies)} experiments in batch (Parallel)...")
            s2_start = time.time()
            results, diagnostic_status = evaluate_v12_batch(policies, df, regime.label, n_jobs)
            s2_duration = time.time() - s2_start
            
            # [V19] Minimum Valid Strategy Safeguard
            valid_count = len([r for r in results if r.score > config.EVAL_SCORE_MIN])
            if valid_count < 1:
                logger.warning("!!! [CRITICAL] No valid strategies found in batch. Action Space might be failing.")
                # Emergency: Force high exploration next batch if complete failure
                eps_manager.reset(force_val=1.0)
            
            # [V19] Reward Variance Watchdog (Learning Loop Disconnect Prevention)
            # Check if all rewards are identical (e.g. all -50.0). This kills D3QN learning.
            scores = [r.score for r in results]
            if len(scores) > 5:
                variance = np.var(scores)
                if variance < 1e-4:
                     logger.warning(f"!!! [WATCHDOG] Reward Variance too low ({variance:.5f}). Learning may stall. Check feature pipeline or gates.")
                     diagnostic_status['status'] = "COLLAPSED"

            # Instrument Stage 2 (Detailed)
            p_rate = valid_count / len(results) if results else 0
            instrumentation.record_stage2(s2_duration, p_rate)
            
            # [V14] Self-Healing: 진단 결과에 따라 정책 조정 (Epsilon 재가열 등)
            agent.adjust_policy(diagnostic_status)
            
            # [V15] Auto-Tuning (Reward Weights, etc.)
            from src.l3_meta.auto_tuner import get_auto_tuner
            tuner = get_auto_tuner()
            tuner.process_diagnostics(diagnostic_status, {})

            # 3. [V11.3] Diversity Selection & Persistence
            from src.l1_judge.diversity import select_diverse_top_k
            
            # Succeeded experiments only (Score above min)
            valid_results = [r for r in results if r.score > config.EVAL_SCORE_MIN]
            
            # Diverse selection among valid candidates
            diverse_results, diversity_info = select_diverse_top_k(
                valid_results, 
                k=config.DIVERSITY_K,
                jaccard_th=config.DIVERSITY_JACCARD_TH,
                param_th=config.DIVERSITY_PARAM_DIST_TH
            )
            
            # Instrument Quality & Diversity
            if valid_results:
                scs = [r.score for r in valid_results]
                instrumentation.record_quality(max(scs), np.mean(scs), np.median(scs))
            
            if diversity_info:
                instrumentation.record_diversity(
                    diversity_info.get("avg_jaccard", 0.0),
                    diversity_info.get("collision_rate", 0.0)
                )
            
            # 4. [V16] One-Pass Persistence (Save all results for 'Include Rejected')
            try:
                from src.l3_meta.eagl import get_eagl_engine
                eagl = get_eagl_engine()
                for res in results:
                    aos = eagl.calculate_aos(res)
                    res.aos_score = aos
                    res.policy_spec.aos_score = aos # Sync back to spec
                    
                    viable, reason = eagl.evaluate_viability(res)
                    res.is_economically_viable = viable
                    res.viability_reason = reason
                    
                    # Update CRM
                    eagl.update_policy_status(res.policy_spec, success=(res.score > config.EVAL_SCORE_MIN))
                
                saved = persist_best_samples(repo, results, df, existing_ids=existing_ids)
                counter += saved
            except Exception as e:
                import traceback
                logger.error(f"Failed to persist or EAGL update: {e}\n{traceback.format_exc()}")
                instrumentation.record_exception(type(e).__name__)
            batch_rewards = [] # Initialize batch_rewards for curriculum and reporting
            diverse_ids = {r.policy_spec.spec_id for r in diverse_results} # Get IDs of diverse results
            for res in results: # Iterate over all results, not just diverse ones
                from src.shared.metrics import metrics_to_legacy_dict, aggregate_windows
                
                # Default metrics for logging
                status_icon = "OK" if res.score > config.EVAL_SCORE_MIN else "NO"
                t1_share = 0.0
                
                # Build metrics dict for reporting and potential learning
                m_dict = {}
                if res.best_sample:
                    m_dict = metrics_to_legacy_dict(
                        aggregate_windows([res.best_sample.metrics], eval_score_override=res.score)
                    )
                    m_dict["trade_logic"] = res.best_sample.core.get("trade_logic", {})
                    m_dict["is_rejected"] = res.score <= config.EVAL_SCORE_MIN
                    t1_share = m_dict.get('top1_share', 0.0)
                    
                    # Only learn from diverse winners & NOT an error
                    is_error = res.module_key in ("ERROR", "FEATURE_MISSING", "INVALID_SPEC")
                    if res.policy_spec.spec_id in diverse_ids and not is_error:
                        batch_rewards.append((res.score, res.policy_spec, m_dict))
                    
                    if is_error:
                        # [Step 7] Log systemic error to ledger
                        error_type = res.module_key
                        error_msg = res.metadata.get("error", "Unknown systemic error")
                        logger.error(f"  [SYSTEM_ERROR] {error_type} for {res.policy_spec.template_id}: {error_msg}")
                        # Record specifically for UI/Dashboard
                        diag_file = Path(config.LEDGER_DIR) / "system_errors.jsonl"
                        with open(diag_file, "a") as f:
                            f.write(json.dumps({
                                "timestamp": pd.Timestamp.now().isoformat(),
                                "spec_id": res.policy_spec.spec_id,
                                "type": error_type,
                                "message": str(error_msg)
                            }) + "\n")
                elif res.module_key in ("ERROR", "FEATURE_MISSING", "INVALID_SPEC"):
                     # Evaluation itself crashed hard or spec failed
                     err_msg = res.metadata.get("error", "Hard crash")
                     logger.error(f"  [SYSTEM_FAILURE] Batch element failed ({res.module_key}): {err_msg}")
                
                # [V18] Log detailed reward breakdown
                status_icon = "OK" if res.score > config.EVAL_SCORE_MIN else "NO"
                t1_share = 0.0
                breakdown_str = ""
                
                # Calculate detailed breakdown
                if m_dict:
                    t1_share = m_dict.get('top1_share', 0.0)
                    try:
                        shaper = get_reward_shaper()
                        bd = shaper.compute_breakdown(m_dict)
                        # Format: [Ret: 1.2, RR: 0.5, Trd: 0.8, Reg: 1.0]
                        breakdown_str = (
                            f"Ret:{bd.return_component:>.1f} "
                            f"RR:{bd.rr_component:>.1f} "
                            f"Trd:{bd.trades_component:>.1f} "
                            f"Reg:{bd.regime_trade_component:>.1f}"
                        )
                    except Exception as e:
                        logger.error(f"  [ERROR] Breakdown computation failed: {e}")
                        breakdown_str = "Breakdown Error"
                
                logger.info(
                    f"  [{results.index(res) + 1}] {status_icon} {res.policy_spec.template_id:<20} | Score: {res.score:>6.2f} | AOS: {res.aos_score:.2f} | {breakdown_str}"
                )
            
            # 5. [V11] 배치 학습 (Parallel Mode)
            if batch_rewards:
                for reward, policy, metrics in batch_rewards:
                    agent.learn(reward, regime, policy, metrics=metrics)
                    
                    # [V11] Curriculum 결과 기록
                    if curriculum and metrics:
                        shaper = get_reward_shaper()
                        breakdown = shaper.compute_breakdown(metrics)
                        passed, reason = curriculum.evaluate_against_current(
                            total_return_pct=metrics.get("total_return_pct", 0.0),
                            trades_per_year=metrics.get("trades_per_year", 0.0),
                            mdd_pct=metrics.get("mdd_pct", 0.0),
                            win_rate=metrics.get("win_rate", 0.0),
                            alpha=breakdown.alpha,
                            profit_factor=metrics.get("profit_factor", 1.0),
                        )
                        status_change = curriculum.record_result(passed, metrics)
                        if status_change.get("promoted"):
                            logger.info(f"  >>> [Curriculum] PROMOTED to Stage {status_change['stage_after']}! Triggering Exploration Reset.")
                            eps_manager.reset(force_val=0.7) # [V12.3] Reignite learning upon stage up

            # C-1 Complete
            logger.info(f"  >>> [Parallel] Batch Complete. {len(batch_rewards)} diverse strategies selected.")
            
        else:
            # ========================================
            # C-2. 순차 실행 모드 (기존 방식)
            # ========================================
            logger.info(f"  >>> [Sequential] Starting {n_jobs} Experiments (Regime: {regime.label})...")
            saved_total, results = _run_batch_sequential(
                agent, df, regime, history, repo, n_jobs
            )
            counter += saved_total
            
            # Populate batch_rewards for reporting consistency
            batch_rewards = []
            for res in results:
                from src.shared.metrics import metrics_to_legacy_dict, aggregate_windows
                if res.best_sample:
                    m_dict = metrics_to_legacy_dict(
                        aggregate_windows([res.best_sample.metrics], eval_score_override=res.score)
                    )
                    batch_rewards.append((res.score, res.policy_spec, m_dict))
            
            policies = [r.policy_spec for r in results]
            valid_results = [r for r in results if r.score > config.EVAL_SCORE_MIN]
            diverse_results = list(valid_results)
            diagnostic_status = {"status": "OK"}
            logger.info(f"  >>> [Sequential] Batch Complete. {len(results)} experiments run.")

        # ==========================================================================================
        # [V16] BATCH INSTRUMENTATION (Common)
        # ==========================================================================================
        if batch_rewards:
            trades = [r[2].get("n_trades", 0) for r in batch_rewards if r[2]]
            zero_ratio = len([t for t in trades if t == 0]) / len(trades) if trades else 0
            instrumentation.record_trades(np.mean(trades) if trades else 0, zero_ratio)
        
        if valid_results:
            scs = [r.score for r in valid_results if hasattr(r, 'score')]
            if scs:
                instrumentation.record_quality(max(scs), np.mean(scs), np.median(scs))
        
        # ==========================================================================================
        # [V15] UNIFIED BATCH REPORT
        # ==========================================================================================
        batch_duration = time.time() - batch_start_time
        if curriculum:
            c_info = curriculum.get_stage_info()
        else:
            c_info = {
                "description": "CURRICULUM_DISABLED",
                "current_stage": 0,
                "stage_passes": 0,
                "threshold_to_next": 0,
                "target_return_pct": 0,
            }
        e_snap = eps_manager.snapshot()
        cache_stats = get_feature_cache().stats
        
        # Best result in this batch
        best_sharpe = max([r[2].get('sharpe', 0) for r in batch_rewards if len(r) > 2 and r[2]] + [0])
        
        diag_str = diagnostic_status.get('status', 'SOFT') if isinstance(diagnostic_status, dict) else str(diagnostic_status)
        
        report = []
        report.append("=" * 100)
        report.append(f"[{'BATCH #' + str(batch_idx):^15}] Time: {batch_duration:4.1f}s | Status: {diag_str:<10} | Cache Hit: {cache_stats['hit_rate_pct']:>3}%")
        report.append("-" * 100)
        report.append(f"[SYSTEM] Regime: {regime.label:<12} | Eps: {e_snap['epsilon']:<6.4f} | Steps: {e_snap['step_count']:<5} | Reheat: {e_snap['last_reheat']}")
        report.append(f"[STAGE ] {c_info['description']} ({c_info['current_stage']}) | Pass: {c_info['stage_passes']:>2} / {c_info['threshold_to_next']:>2} to Next | Target: {c_info['target_return_pct']}%")
        report.append(f"[EVAL  ] Total: {len(policies):>2} | Valid: {len(valid_results):>2} | Diverse: {len(diverse_results):>2} | Best Sharpe: {best_sharpe:>5.2f}")
        
        # [V18] LogicTree Diagnostics KPI
        from src.shared.logic_tree_diagnostics import get_and_reset_diagnostics
        lt_diag = get_and_reset_diagnostics(batch_id=f"batch_{batch_idx}")
        if lt_diag.total_condition_evals > 0:
            status_icon = "✓" if lt_diag.is_healthy else "⚠"
            report.append(
                f"[TREE  ] {status_icon} Match: {lt_diag.match_rate:.1%} | "
                f"Direct: {lt_diag.matched_direct} | Fuzzy: {lt_diag.matched_fuzzy} | "
                f"Ambig: {lt_diag.ambiguous} | Unmatch: {lt_diag.unmatched}"
            )
            if lt_diag.unmatched_keys:
                report.append(f"[TREE  ] Missing keys: {lt_diag.unmatched_keys[:5]}")
        
        if hasattr(tuner, 'current_weights'):
            tw = tuner.current_weights
            report.append(f"[TUNER ] W_CAGR: {tw['reward_cagr']:.2f} | W_MDD: {tw['reward_mdd']:.2f} | W_Complex: {tw['complexity_penalty']:.2f}")
            
        report.append("=" * 100)
        
        for line in report:
            logger.info(line)

        # Apply decay after report
        eps_manager.apply_step()
        
        # [V16] End Batch Instrumentation
        instrumentation.end_batch()

        # F. Iterate
        time.sleep(sleep_sec)


if __name__ == "__main__":
    infinite_loop()

