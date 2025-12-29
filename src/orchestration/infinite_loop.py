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
from src.orchestration.evaluation import (
    evaluate_stage,
    evaluate_v12_batch,
    persist_best_samples,
    ModuleResult,
    prefilter_policies_by_entry_rate,
)
from src.data.loader import DataLoader
from src.contracts import PolicySpec
from src.shared.instrumentation import get_instrumentation
from src.shared.observability import (
    build_episode_summary,
    build_batch_report,
    log_batch,
    log_invalid_actions,
    load_recent_batch_reports,
    detect_deadlock,
)
from src.orchestration.stage_controller import StageController
from src.orchestration.regression_monitor import RegressionMonitor

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
    stage_id: Optional[int] = None,
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
        stage_id = int(stage_id or getattr(config, "CURRICULUM_CURRENT_STAGE", 1))
        config.CURRICULUM_CURRENT_STAGE = stage_id
        results, status = evaluate_stage([policy], df, "full", regime_label, n_jobs=1, stage_id=stage_id)
        
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
    stage_id: Optional[int] = None,
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
            stage_id=stage_id,
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
    stage_id: Optional[int] = None,
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
            stage_id = int(stage_id or getattr(config, "CURRICULUM_CURRENT_STAGE", 1))
            config.CURRICULUM_CURRENT_STAGE = stage_id
            results, status = evaluate_v12_batch([pol], df, regime.label, n_jobs=1, stage_id=stage_id)
            
            # 3. 저장
            existing_ids = {r.policy_spec.spec_id for r in history}
            saved = persist_best_samples(repo, results, df, existing_ids=existing_ids)
            saved_total += saved
            
            # 4. 즉시 학습
            if results:
                from src.shared.metrics import metrics_to_legacy_dict, aggregate_windows
                res = results[0]
                reward = res.score
                if res.reward_breakdown and isinstance(res.reward_breakdown, dict):
                    reward = float(res.reward_breakdown.get("total", reward))
                
                m_dict = {}
                if res.best_sample:
                    m_dict = metrics_to_legacy_dict(
                        aggregate_windows([res.best_sample.metrics], eval_score_override=res.score)
                    )
                    m_dict["trade_logic"] = res.best_sample.core.get("trade_logic", {})
                    m_dict["is_rejected"] = res.score <= config.EVAL_SCORE_MIN
                
                agent.learn(reward, regime, pol, metrics=m_dict)
                batch_results.append(res)
                
                status_icon = "통과" if reward > config.EVAL_SCORE_MIN else "실패"
                logger.info(
                    f"  [{i+1}] {status_icon} {pol.template_id:<20} | 점수: {reward:>6.3f}"
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
                    status_change = curriculum.record_result(
                        passed,
                        m_dict,
                        allow_stage_change=not getattr(config, "STAGE_AUTO_ENABLED", True),
                    )
                    if status_change.get("promoted"):
                        logger.info(f"  >>> [커리큘럼] 스테이지 {status_change['stage_after']} 승격.")
        except Exception as e:
            exc_type = type(e).__name__
            from src.shared.instrumentation import get_instrumentation
            get_instrumentation().record_exception(exc_type)
            logger.error(f"!!! [순차 오류] {e}", exc_info=True)
    
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
    max_batches = int(getattr(config, "MAX_BATCHES", 0))
    sleep_sec = sleep_interval if sleep_interval is not None else config.SLEEP_INTERVAL

    # 1. Setup
    logger.info(">>> [시스템] 엔진 초기화 중...")
    
    # DI: 싱글톤 레지스트리 사용 (중복 초기화 방지)
    registry = get_registry()
    registry.warmup()  # [Optim] 핸들러 사전 컴파일 (메인 프로세스 1회)
    
    repo = LedgerRepo(l_path)
    agent = MetaAgent(registry, repo)
    detector = RegimeDetector()
    eps_manager = get_epsilon_manager()
    instrumentation = get_instrumentation()
    stage_controller = StageController()
    regression_monitor = RegressionMonitor()
    
    # 병렬 설정
    n_jobs = config.PARALLEL_BATCH_SIZE or multiprocessing.cpu_count()
    parallel_mode = config.PARALLEL_ENABLED
    policy_batch_size = int(getattr(config, "BATCH_POLICY_COUNT", 0))
    if policy_batch_size <= 0:
        policy_batch_size = n_jobs
    
    # D3QN 모드 확인
    d3qn_mode = config.D3QN_ENABLED
    logger.info(
        f">>> [시스템] 병렬 모드: {parallel_mode} | 워커: {n_jobs} | 정책: {policy_batch_size} | D3QN: {d3qn_mode}"
    )
    
    # [V11] Curriculum Controller 초기화
    curriculum = get_curriculum_controller() if config.CURRICULUM_ENABLED else None
    if curriculum:
        stage_info = curriculum.get_stage_info()
        logger.info(
            f">>> [커리큘럼] 스테이지 {stage_info['current_stage']}: "
            f"{stage_info['description']} | "
            f"통과 {stage_info['stage_passes']}/{stage_info['threshold_to_next']}"
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
        
        logger.info(f">>> [데이터] 로더 초기화: {target} ({source})")
        if source == "binance":
            from src.data.binance_loader import fetch_btc_training_data
            df = fetch_btc_training_data()
        else:
            loader = DataLoader(target_ticker=target, start_date=config.DATA_START_DATE)
            df = loader.fetch_all()
            
        if df.empty:
            raise ValueError(f"Fetched data is empty for {target}")
            
        logger.info(f">>> [데이터] 로드 완료: {len(df)} bars")
        
        # [V11.2] Pre-calculate Regime Labels for backtest validation
        logger.info(">>> [시스템] 레짐 사전 계산 중...")
        df["regime_label"] = detector.detect_series(df)
        
        agent.set_market_data(df)
    except Exception as e:
        logger.error(f"!!! [오류] 데이터 로드 실패: {e}", exc_info=True)
        return

    # Resume Counter from History
    try:
        existing_records = repo.load_records()
        counter = len(existing_records)
        logger.info(f">>> [시스템] 기존 실험 {counter}건 발견. #{counter + 1}부터 재개.")
    except Exception as e:
        counter = 0
        logger.warning(f">>> [시스템] 기록 로드 실패로 새로 시작: {e}")

    batch_idx = 0
    
    # 3. The Loop
    while True:
        batch_start_time = time.time()
        
        # Performance Based Pruning (Maintenance)
        if max_exps > 0 and counter > max_exps:
            pruned = repo.prune_experiments(keep_n=max_exps)
            if pruned > 0:
                logger.info(f">>> [시스템] 하위 성능 {pruned}건 정리 (풀 크기 {max_exps} 유지)")

        batch_idx += 1
        
        # [V16] Deterministic Batch Seed
        # Formula: Base Seed (e.g. 2025) + Batch Index
        batch_seed = 2025 + batch_idx
        instrumentation.start_batch(batch_idx, batch_seed, policy_batch_size)

        # Defaults for batch-level reporting to avoid undefined vars
        policies = []
        results = []
        valid_results = []
        diverse_results = []
        batch_rewards = []
        diagnostic_status = {"status": "EMPTY"}
        tuner = None
        diversity_info = {}

        # A. Detect Regime (Situation Awareness)
        regime = detector.detect(df)
        
        # [Alpha-Power V1] Market Context Modulation
        agent.apply_market_context_modulation(regime)

        # B. Load History (Memory)
        history = repo.load_records()
        existing_ids = {r.policy_spec.spec_id for r in history}
        if curriculum:
            current_stage = int(curriculum.current_stage)
        else:
            current_stage = int(getattr(config, "CURRICULUM_CURRENT_STAGE", 1))
        config.CURRICULUM_CURRENT_STAGE = current_stage

        # logger.info("-" * 90) removed to replace with better header
        
        if parallel_mode:
            # ========================================
            # C-1. 병렬 실행 모드
            # ========================================
            logger.info(f"  >>> [병렬] 정책 생성 {policy_batch_size}개 (레짐: {regime.label})")
            
            # 1. 정책 배치 생성 (Diversity-aware Batch)
            policies = agent.propose_batch(regime, history, policy_batch_size, seed=batch_seed)

            prefilter_stats = None
            if getattr(config, "ENTRY_HIT_RATE_FILTER_ENABLED", False):
                target_keep = int(getattr(config, "PREFILTER_TARGET_KEEP", policy_batch_size))
                if target_keep <= 0:
                    target_keep = policy_batch_size
                min_keep = int(getattr(config, "PREFILTER_MIN_KEEP", 0))
                max_attempts = int(getattr(config, "PREFILTER_MAX_ATTEMPTS", 0))

                kept: List[PolicySpec] = []
                reject_reason_counts = {}
                attempts_total = 0
                total_seen = 0
                mean_rate_sum = 0.0
                last_bounds = {}

                while attempts_total < max_attempts and len(kept) < target_keep:
                    needed = target_keep - len(kept)
                    batch_need = min(needed, policy_batch_size)
                    candidates = agent.propose_batch(
                        regime,
                        history,
                        batch_need,
                        seed=batch_seed + attempts_total,
                    )
                    attempts_total += len(candidates)

                    kept_batch, stats = prefilter_policies_by_entry_rate(
                        candidates,
                        df,
                        stage_id=current_stage,
                    )
                    total_seen += int(stats.get("total", 0))
                    mean_rate_sum += float(stats.get("mean_rate", 0.0)) * float(stats.get("total", 0))
                    last_bounds = {
                        "min_rate": stats.get("min_rate"),
                        "max_rate": stats.get("max_rate"),
                    }
                    kept.extend(kept_batch)
                    for reason, count in (stats.get("reject_reason_counts") or {}).items():
                        reject_reason_counts[reason] = reject_reason_counts.get(reason, 0) + int(count)

                prefilter_stats = {
                    "attempts_total": attempts_total,
                    "kept_total": len(kept),
                    "target_keep": target_keep,
                    "min_keep": min_keep,
                    "max_attempts": max_attempts,
                    "reject_reason_counts": reject_reason_counts,
                    "mean_rate": (mean_rate_sum / total_seen) if total_seen > 0 else 0.0,
                    "min_rate": last_bounds.get("min_rate", 0.0),
                    "max_rate": last_bounds.get("max_rate", 1.0),
                }

                if len(kept) >= min_keep:
                    policies = kept[:target_keep]
                else:
                    diagnostic_status = {"status": "PREFILTER_STUCK", "prefilter": prefilter_stats}
                    policies = []
                    results = []
                    valid_results = []
                    diverse_results = []
                    batch_rewards = []
            
            # 2. 병렬 평가 (V14 Optimized)
            logger.info(f"  >>> [평가] 배치 평가 {len(policies)}개 (병렬)")
            s2_start = time.time()
            if policies:
                results, diagnostic_status = evaluate_v12_batch(
                    policies,
                    df,
                    regime.label,
                    n_jobs,
                    stage_id=current_stage,
                    prefiltered=bool(prefilter_stats),
                )
                if prefilter_stats and isinstance(diagnostic_status, dict):
                    diagnostic_status["prefilter"] = prefilter_stats
            s2_duration = time.time() - s2_start
            
            from src.l1_judge.evaluator import compute_gate_diagnostics

            def _is_execution_success(res: ModuleResult) -> bool:
                if not res.best_sample or not res.best_sample.metrics:
                    return False
                if res.module_key in ("ERROR", "FEATURE_MISSING", "INVALID_SPEC"):
                    return False
                if getattr(res.best_sample.metrics, "is_rejected", False):
                    return False
                return True

            gate_diag_by_id = {}
            learning_candidates = []
            for res in results:
                if not _is_execution_success(res):
                    continue
                metrics = res.best_sample.metrics
                gate_diag = compute_gate_diagnostics(metrics)
                gate_diag_by_id[res.policy_spec.spec_id] = gate_diag
                learning_candidates.append(res)

            valid_results = [
                r for r in learning_candidates
                if gate_diag_by_id.get(r.policy_spec.spec_id)
                and gate_diag_by_id[r.policy_spec.spec_id].approval_pass
            ]

            # [V19] Minimum Valid Strategy Safeguard
            valid_count = len(valid_results)
            if valid_count < 1:
                logger.warning("!!! [치명] 유효 전략 없음. 액션 공간이 비활성일 수 있음.")
                # Emergency: Force high exploration next batch if complete failure
                eps_manager.reset(force_val=1.0)
            
            # [V19] Reward Variance Watchdog (Learning Loop Disconnect Prevention)
            # Check if all rewards are identical (e.g. all -50.0). This kills D3QN learning.
            scores = [r.score for r in results]
            if len(scores) > 5:
                variance = np.var(scores)
                if variance < 1e-4:
                     logger.warning(f"!!! [감시] 보상 분산이 너무 낮음 ({variance:.5f}). 학습 정체 가능.")
                     diagnostic_status['status'] = "COLLAPSED"

            # Instrument Stage 2 (Detailed)
            
            # [V14] Self-Healing: 진단 결과에 따라 정책 조정 (Epsilon 재가열 등)
            agent.adjust_policy(diagnostic_status)
            
            # [V15] Auto-Tuning (Reward Weights, etc.)
            from src.l3_meta.auto_tuner import get_auto_tuner
            tuner = get_auto_tuner()
            extra_info = {"batch_id": batch_idx}
            current_metrics = getattr(instrumentation, "current_metrics", None)
            if current_metrics is not None:
                extra_info["pass_rate_s1"] = float(getattr(current_metrics, "pass_rate_s1", 0.0))
                extra_info["pass_rate_s2"] = float(getattr(current_metrics, "pass_rate_s2", 0.0))
                extra_info["exception_count"] = int(sum(getattr(current_metrics, "exceptions", {}).values()))
            tuner.process_diagnostics(diagnostic_status, extra_info)

            # 3. [V11.3] Diversity Selection & Persistence
            from src.l1_judge.diversity import select_diverse_top_k

            soft_gate_k = int(getattr(config, "SELECTION_SOFT_GATE_TOP_K", 0))
            if soft_gate_k < 0:
                soft_gate_k = 0
            soft_gate_elites = []
            if soft_gate_k > 0 and valid_results:
                def _soft_gate_score(res):
                    diag = gate_diag_by_id.get(res.policy_spec.spec_id)
                    return diag.soft_gate_score if diag else float("-inf")

                soft_gate_elites = sorted(
                    valid_results,
                    key=_soft_gate_score,
                    reverse=True,
                )[:soft_gate_k]

            # Diverse selection among learning candidates
            diverse_results, diversity_info = select_diverse_top_k(
                valid_results,
                k=config.DIVERSITY_K,
                jaccard_th=config.DIVERSITY_JACCARD_TH,
                param_th=config.DIVERSITY_PARAM_DIST_TH,
                gate_diag_map=gate_diag_by_id,
                seed=soft_gate_elites,
            )

            gate_focus = None
            if gate_diag_by_id:
                gate_counts = {}
                for diag in gate_diag_by_id.values():
                    if diag.nearest_gate != "PASS":
                        gate_counts[diag.nearest_gate] = gate_counts.get(diag.nearest_gate, 0) + 1
                if gate_counts:
                    gate_focus = max(gate_counts, key=gate_counts.get)
            if gate_focus and hasattr(agent, "update_mutation_gate_focus"):
                agent.update_mutation_gate_focus(gate_focus)
            
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
                logger.error(f"저장 또는 EAGL 업데이트 실패: {e}\n{traceback.format_exc()}")
                instrumentation.record_exception(type(e).__name__)
            batch_rewards = [] # Initialize batch_rewards for curriculum and reporting
            diverse_ids = {r.policy_spec.spec_id for r in diverse_results} # Get IDs of diverse results
            template_summary = {}
            summary_order = []
            for res in results: # Iterate over all results, not just diverse ones
                from src.shared.metrics import metrics_to_legacy_dict, aggregate_windows
                
                # Default metrics for logging
                status_icon = "통과" if res.score > config.EVAL_SCORE_MIN else "실패"
                
                # Build metrics dict for reporting and potential learning
                m_dict = {}
                if res.best_sample:
                    m_dict = metrics_to_legacy_dict(
                        aggregate_windows([res.best_sample.metrics], eval_score_override=res.score)
                    )
                    m_dict["trade_logic"] = res.best_sample.core.get("trade_logic", {})
                    m_dict["is_rejected"] = res.score <= config.EVAL_SCORE_MIN
                    gate_diag = gate_diag_by_id.get(res.policy_spec.spec_id)
                    if gate_diag:
                        m_dict["soft_gate_score"] = float(gate_diag.soft_gate_score)
                        m_dict["nearest_gate"] = gate_diag.nearest_gate
                        m_dict["approval_pass"] = gate_diag.approval_pass
                    t1_share = m_dict.get('top1_share', 0.0)
                    
                    # Only learn from diverse winners & NOT an error
                    is_error = res.module_key in ("ERROR", "FEATURE_MISSING", "INVALID_SPEC")
                    if res.policy_spec.spec_id in diverse_ids and not is_error:
                        reward_total = res.score
                        if res.reward_breakdown and isinstance(res.reward_breakdown, dict):
                            reward_total = float(res.reward_breakdown.get("total", reward_total))
                        batch_rewards.append((reward_total, res.policy_spec, m_dict))
                    
                    if is_error:
                        # [Step 7] Log systemic error to ledger
                        error_type = res.module_key
                        error_msg = res.metadata.get("error", "Unknown systemic error")
                        logger.error(f"  [시스템오류] {error_type} ({res.policy_spec.template_id}): {error_msg}")
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
                     logger.error(f"  [시스템실패] 배치 요소 실패 ({res.module_key}): {err_msg}")
                
                # [V18] Log detailed reward breakdown
                breakdown_str = ""
                
                # Calculate detailed breakdown
                if m_dict:
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
                        logger.error(f"  [오류] 보상 분해 계산 실패: {e}")
                        breakdown_str = "Breakdown Error"
                # [V18] Improved status line with rejection reason
                rej_reason = m_dict.get("rejection_reason", "N/A") if res.score <= config.EVAL_SCORE_MIN else ""
                rej_str = f" | 사유: {rej_reason}" if rej_reason and rej_reason != "PASS" else ""

                template_id = res.policy_spec.template_id
                if template_id not in template_summary:
                    template_summary[template_id] = {
                        "count": 0,
                        "best_score": res.score,
                        "best_aos": res.aos_score,
                        "status": status_icon,
                        "breakdown": breakdown_str,
                        "rej_reason": rej_str,
                    }
                    summary_order.append(template_id)

                summary = template_summary[template_id]
                summary["count"] += 1
                if res.score >= summary["best_score"]:
                    summary["best_score"] = res.score
                    summary["best_aos"] = res.aos_score
                    summary["status"] = status_icon
                    summary["breakdown"] = breakdown_str
                    summary["rej_reason"] = rej_str

            if template_summary:
                logger.info("  >>> [요약] 템플릿별 결과")
                for template_id in summary_order:
                    summary = template_summary[template_id]
                    breakdown = summary["breakdown"]
                    reason = summary["rej_reason"]
                    detail = f" | {breakdown}" if breakdown else ""
                    logger.info(
                        f"  - {template_id:<20} x{summary['count']} | {summary['status']} | "
                        f"최고점수: {summary['best_score']:>6.2f} | AOS: {summary['best_aos']:.2f}{detail}{reason}"
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
                        status_change = curriculum.record_result(
                            passed,
                            metrics,
                            allow_stage_change=not getattr(config, "STAGE_AUTO_ENABLED", True),
                        )
                        if status_change.get("promoted"):
                            logger.info(f"  >>> [커리큘럼] 스테이지 {status_change['stage_after']} 승격. 탐색 리셋.")
                            eps_manager.reset(force_val=0.7) # [V12.3] Reignite learning upon stage up

            # C-1 Complete
            logger.info(f"  >>> [병렬] 배치 완료. 다양성 전략 {len(batch_rewards)}개 선택.")
            
        else:
            # ========================================
            # C-2. 순차 실행 모드 (기존 방식)
            # ========================================
            logger.info(f"  >>> [순차] 실험 시작 {policy_batch_size}개 (레짐: {regime.label})")
            saved_total, results = _run_batch_sequential(
                agent, df, regime, history, repo, policy_batch_size, stage_id=current_stage
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
                    reward_total = res.score
                    if res.reward_breakdown and isinstance(res.reward_breakdown, dict):
                        reward_total = float(res.reward_breakdown.get("total", reward_total))
                    batch_rewards.append((reward_total, res.policy_spec, m_dict))
            
            policies = [r.policy_spec for r in results]
            from src.l1_judge.evaluator import compute_gate_diagnostics
            valid_results = [
                r for r in results
                if r.best_sample
                and r.best_sample.metrics
                and r.module_key not in ("ERROR", "FEATURE_MISSING", "INVALID_SPEC")
            ]
            valid_results = [
                r for r in valid_results
                if not getattr(r.best_sample.metrics, "is_rejected", False)
                and compute_gate_diagnostics(r.best_sample.metrics).approval_pass
            ]
            diverse_results = list(valid_results)
            diagnostic_status = {"status": "OK"}
            logger.info(f"  >>> [순차] 배치 완료. 실험 {len(results)}개 실행.")

        # ==========================================================================================
        # [V16] BATCH INSTRUMENTATION (Common)
        # ==========================================================================================
        stage_cfg = config.CURRICULUM_STAGES.get(current_stage, {})
        from src.l1_judge.diversity import compute_selection_score, compute_robustness_score
        from src.l1_judge.evaluator import compute_gate_diagnostics
        valid_ids = {r.policy_spec.spec_id for r in valid_results}
        selected_ids = {r.policy_spec.spec_id for r in diverse_results}
        gate_diag_map = {}
        for res in results:
            metrics = getattr(getattr(res, "best_sample", None), "metrics", None)
            if metrics is None:
                continue
            gate_diag_map[res.policy_spec.spec_id] = compute_gate_diagnostics(metrics)

        episode_summaries = []
        from src.shared.metrics import TradeStats, EquityStats, WindowMetrics
        for res in results:
            if res.best_sample and res.best_sample.metrics:
                metrics = res.best_sample.metrics
                core = res.best_sample.core or {}
                bt_result = core.get("bt_result")
            else:
                metrics = WindowMetrics(
                    window_id="EPISODE",
                    trades=TradeStats(),
                    equity=EquityStats(),
                    bars_total=len(df),
                )
                bt_result = None

            summary = build_episode_summary(
                policy_id=res.policy_spec.spec_id,
                stage=res.stage,
                metrics=metrics,
                bt_result=bt_result,
                reward_breakdown=res.reward_breakdown,
                eval_score=res.score,
                module_key=res.module_key,
                batch_id=batch_idx,
            )

            policy_id = res.policy_spec.spec_id
            gate_pass = bool(summary.get("gate", {}).get("gate_pass", False))
            selection_status = {
                "gate_pass": gate_pass,
                "valid": policy_id in valid_ids,
                "selected": policy_id in selected_ids,
            }

            performance = float(summary.get("reward_total", summary.get("eval_score", 0.0)))
            gate_diag = gate_diag_map.get(policy_id)
            if gate_diag is not None:
                progress = float(getattr(gate_diag, "soft_gate_score", 0.0))
                robustness = compute_robustness_score(res, stage_cfg)
                selection_score = compute_selection_score(res, gate_diag, stage_cfg)
            else:
                progress = float(summary.get("gate", {}).get("soft_gate_score", 0.0))
                robustness = 0.0
                selection_score = performance

            summary["selection_status"] = selection_status
            summary["selection_components"] = {
                "performance": performance,
                "progress": progress,
                "robustness": robustness,
                "selection_score": selection_score,
            }

            episode_summaries.append(summary)

        batch_report = build_batch_report(batch_idx, episode_summaries)
        batch_report["stage_snapshot"] = {
            "stage_id": int(current_stage),
            "source": "curriculum",
        }
        if isinstance(diagnostic_status, dict):
            prefilter = diagnostic_status.get("prefilter")
            if prefilter:
                batch_report["policy_prefilter"] = prefilter
        if isinstance(diagnostic_status, dict):
            eval_usage = diagnostic_status.get("eval_usage")
            if eval_usage:
                batch_report["eval_usage"] = eval_usage
        wf_pass_rate = 0.0
        wf_alpha_min_mean = 0.0
        wf_alpha_std_mean = 0.0
        wf_samples = []
        stage_cfg = config.CURRICULUM_STAGES.get(config.CURRICULUM_CURRENT_STAGE, {})
        alpha_floor = float(getattr(stage_cfg, "alpha_floor", -10.0)) if stage_cfg else -10.0
        for res in results:
            if not res.window_results:
                continue
            alphas = [w.avg_alpha for w in res.window_results if w is not None]
            if not alphas:
                continue
            wf_samples.append({
                "pass": all(a >= alpha_floor for a in alphas),
                "min_alpha": float(min(alphas)),
                "std_alpha": float(np.std(alphas)) if len(alphas) > 1 else 0.0,
            })
        if wf_samples:
            wf_pass_rate = float(np.mean([1.0 if s["pass"] else 0.0 for s in wf_samples]))
            wf_alpha_min_mean = float(np.mean([s["min_alpha"] for s in wf_samples]))
            wf_alpha_std_mean = float(np.mean([s["std_alpha"] for s in wf_samples]))

        batch_report["wf_pass_rate"] = wf_pass_rate
        batch_report["wf_alpha_min_mean"] = wf_alpha_min_mean
        batch_report["wf_alpha_std_mean"] = wf_alpha_std_mean
        batch_report["diversity_info"] = diversity_info
        history = load_recent_batch_reports(getattr(config, "OBS_DEADLOCK_WINDOW", 20))
        batch_report["deadlock"] = detect_deadlock(history + [batch_report])
        collapse_window = int(getattr(config, "REWARD_COLLAPSE_TREND_WINDOW", 5))
        collapse_min_count = int(getattr(config, "REWARD_COLLAPSE_TREND_MIN_COUNT", 3))
        collapse_history = (history + [batch_report])[-collapse_window:] if collapse_window > 0 else [batch_report]
        collapse_count = sum(
            1 for r in collapse_history if r.get("reward_collapse", {}).get("collapsed", False)
        )
        batch_report["reward_collapse_trend"] = {
            "window": collapse_window,
            "min_count": collapse_min_count,
            "count": int(collapse_count),
            "collapsed": bool(collapse_count >= collapse_min_count),
        }
        failure_mode = regression_monitor.classify_failure_mode(batch_report, history)
        batch_report["failure_mode_decision"] = failure_mode.to_dict()
        regression_monitor.log_failure_mode(failure_mode, batch_report)

        stage_decision = stage_controller.update(batch_report, batch_idx)
        batch_report["stage_transition"] = stage_decision.to_dict()
        if stage_decision.action in ("promote", "demote"):
            logger.info(
                f">>> [Stage] {stage_decision.action.upper()} {stage_decision.stage_before} -> {stage_decision.stage_after} | "
                f"Reasons: {', '.join(stage_decision.reasons) if stage_decision.reasons else 'N/A'}"
            )

        log_invalid_actions(batch_idx, episode_summaries)
        log_batch(batch_report)

        regression_report = regression_monitor.evaluate(
            batch_report=batch_report,
            episode_summaries=episode_summaries,
            diversity_info=diversity_info,
            history=history,
        )
        if regression_report:
            codes = [s.code for s in regression_report.signals if s.triggered]
            logger.warning(f"!!! [퇴행] 감지됨: {', '.join(codes)}")

        if batch_rewards:
            trades = [r[2].get("n_trades", 0) for r in batch_rewards if r[2]]
            zero_ratio = len([t for t in trades if t == 0]) / len(trades) if trades else 0
            instrumentation.record_trades(np.mean(trades) if trades else 0, zero_ratio)

            cycles = [r[2].get("cycle_count", 0) for r in batch_rewards if r[2]]
            zero_cycle_ratio = len([c for c in cycles if c == 0]) / len(cycles) if cycles else 0
            instrumentation.record_cycles(np.mean(cycles) if cycles else 0, zero_cycle_ratio)
        
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
        cycle_hist = batch_report.get("cycle_count_hist", {})
        report.append(
            f"[CYCLE ] Median: {cycle_hist.get('median', 0.0):>5.2f} | "
            f"P10: {cycle_hist.get('p10', 0.0):>5.2f} | "
            f"P90: {cycle_hist.get('p90', 0.0):>5.2f} | "
            f"GatePass: {batch_report.get('gate_pass_rate', 0.0):>5.2f}"
        )
        report.append(
            f"[SINGLE] Rate: {batch_report.get('single_trade_rate', 0.0):>5.2f} | "
            f"EntryRate: {batch_report.get('single_trade_entry_rate_mean', 0.0):>6.4f}"
        )
        decision = batch_report.get("failure_mode_decision", {}) or {}
        decision_action = decision.get("action", "continue")
        decision_reasons = decision.get("reasons", [])
        reasons_str = ", ".join(decision_reasons[:2]) if decision_reasons else "n/a"
        report.append(f"[DECIS] {decision_action.upper():<9} | {reasons_str}")

        perf_summary = batch_report.get("performance_summary", {})
        if perf_summary:
            ret = perf_summary.get("total_return_pct", {})
            mdd = perf_summary.get("mdd_pct", {})
            sharpe = perf_summary.get("sharpe", {})
            win_rate = perf_summary.get("win_rate", {})
            trades_year = perf_summary.get("trades_per_year", {})
            report.append(
                f"[PERF ] Ret: {ret.get('mean', 0.0):>6.2f}% (P10 {ret.get('p10', 0.0):>6.2f}, P90 {ret.get('p90', 0.0):>6.2f}) | "
                f"MDD: {mdd.get('mean', 0.0):>6.2f}% | "
                f"Sharpe: {sharpe.get('mean', 0.0):>5.2f} | "
                f"Win: {win_rate.get('mean', 0.0):>5.2f} | "
                f"Trades/Y: {trades_year.get('mean', 0.0):>5.1f}"
            )

        topk_perf = batch_report.get("topk_performance", [])
        if topk_perf:
            topk_lines = []
            for idx, perf in enumerate(topk_perf[:3], start=1):
                topk_lines.append(
                    f"#{idx} score={perf.get('reward_total', 0.0):.1f} "
                    f"ret={perf.get('total_return_pct', 0.0):.1f}% "
                    f"alpha={perf.get('excess_return', 0.0):.1f}% "
                    f"mdd={perf.get('mdd_pct', 0.0):.1f}% "
                    f"trd={perf.get('trade_count', 0)} "
                    f"cyc={perf.get('cycle_count', 0)} "
                    f"gate={perf.get('nearest_gate', 'PASS')}"
                )
            report.append(f"[TOPK ] {' | '.join(topk_lines)}")
        
        # [V18] LogicTree Diagnostics KPI
        from src.shared.logic_tree_diagnostics import get_and_reset_diagnostics
        lt_diag = get_and_reset_diagnostics(batch_id=f"batch_{batch_idx}")
        if lt_diag.total_condition_evals > 0:
            status_icon = "정상" if lt_diag.is_healthy else "경고"
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

        if max_batches > 0 and batch_idx >= max_batches:
            logger.info(f">>> [시스템] 최대 배치 도달 ({max_batches}). 종료.")
            break

        # F. Iterate
        time.sleep(sleep_sec)


if __name__ == "__main__":
    infinite_loop()

