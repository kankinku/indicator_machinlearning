from __future__ import annotations

import json
import time
import copy
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.config import config
from src.contracts import PolicySpec
from src.l1_judge.evaluator import (
    SampleMetrics,
    aggregate_sample_scores,
    build_module_key,
    compute_sample_metrics,
    get_risk_distributions,
    is_violation,
    score_sample,
    normalize_scores,
    sample_risk_params,
    stable_hash,
    validate_sample,
    collect_validation_failures,
)
from src.evaluation.real_evaluator import RealEvaluator
from src.evaluation.fast_filter import FastFilter
from src.orchestration.run_experiment import _generate_features_cached, _run_experiment_core, build_record_and_artifact
from src.evaluation.contracts import WindowResult, BestSample, ModuleResult, EvaluationResult
from src.evaluation.result_store import get_result_store, get_policy_sig
from src.shared.logger import get_logger
from src.shared.instrumentation import get_instrumentation
from src.shared.hashing import generate_policy_id, get_eval_config_signature, hash_dataframe, calculate_sha256
from src.l3_meta.reward_shaper import get_reward_shaper
from src.shared.event_bus import record_event
from src.shared.caching import get_signal_cache, get_backtest_cache
from src.shared.observability import build_episode_summary, log_episode

logger = get_logger("orchestration.evaluation")


class OperationalQACollector:
    """
    [V14] Operational QA Monitoring
    Tracks strategy pass/fail rates and failure taxonomy across a batch.
    """
    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.total_samples = 0
        self.passed_samples = 0
        self.failure_counts = {}
        self.performance_stats = {
            "returns": [],
            "mdds": [],
            "annual_trades": [],
            "zero_trades": 0,
            "scores": []
        }
        self.similarity_stats = {
            "avg_jaccard": 0.0,
            "collision_rate": 0.0
        }

    def collect(self, passed: bool, reason: str, metrics: Optional[SampleMetrics] = None, score: float = config.EVAL_SCORE_MIN):
        self.total_samples += 1
        self.performance_stats["scores"].append(score)
        if passed and metrics:
            self.passed_samples += 1
            self.performance_stats["returns"].append(metrics.equity.total_return_pct)
            self.performance_stats["mdds"].append(metrics.equity.max_drawdown_pct)
            self.performance_stats["annual_trades"].append(metrics.trades.trades_per_year)
            if metrics.trades.trade_count == 0:
                self.performance_stats["zero_trades"] += 1
        else:
            # Extract FAIL_XXX code
            fail_code = reason.split("(")[0].strip() if "(" in reason else reason
            self.failure_counts[fail_code] = self.failure_counts.get(fail_code, 0) + 1
            if fail_code == "FAIL_ZERO_EXPOSURE":
                self.performance_stats["zero_trades"] += 1

    def _get_category(self, fail_code: str) -> str:
        for cat, codes in config.FAILURE_TAXONOMY.items():
            if fail_code in codes:
                return cat
        return "UNKNOWN_ISSUE"

    def similarity_analysis(self, policies: List[PolicySpec]):
        """[V14] Detects redundant strategies in the batch."""
        from src.l1_judge.diversity import calculate_genome_similarity
        if len(policies) < 2:
            return
            
        jaccards = []
        collisions = 0
        n = len(policies)
        
        # Simple sampling if batch is too large
        sample_size = min(n, 20)
        subset = policies[:sample_size]
        
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                sim = calculate_genome_similarity(subset[i], subset[j])
                jaccards.append(sim)
                if sim > 0.8: # Hard threshold for collision detection
                    collisions += 1
                    
        pairs = len(jaccards)
        if pairs > 0:
            self.similarity_stats["avg_jaccard"] = np.mean(jaccards)
            self.similarity_stats["collision_rate"] = collisions / pairs
            
        if self.similarity_stats["collision_rate"] > 0.3:
            logger.warning(f"[QA] 전략 중복 경고: 배치 충돌률 {self.similarity_stats['collision_rate']:.1%}")

    def report(self):
        if self.total_samples == 0:
            logger.warning(f"[QA] 샘플 없음: {self.stage_name}")
            return

        pass_rate = self.passed_samples / self.total_samples
        rej_rate = 1.0 - pass_rate
        
        logger.info(f"[QA] 진단 보고: {self.stage_name.upper()}")
        logger.info(f"[QA] 요약: 통과 {pass_rate:.1%} | 탈락 {rej_rate:.1%} (n={self.total_samples})")
        
        # 1. Failure Taxonomy Grouping
        taxonomy = {}
        if self.failure_counts:
            for code, count in self.failure_counts.items():
                cat = self._get_category(code)
                taxonomy[cat] = taxonomy.get(cat, 0) + count
            
            tax_str = ", ".join([f"{k}: {v}" for k, v in taxonomy.items()])
            logger.info(f"[QA] 실패 분류: {tax_str}")
            
            top_raw = sorted(self.failure_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            raw_str = ", ".join([f"{k}({v})" for k, v in top_raw])
            logger.info(f"[QA] 주요 사유: {raw_str}")

        # 2. Performance Profiling
        avg_tpy = 0.0
        if self.passed_samples > 0:
            avg_ret = np.mean(self.performance_stats["returns"])
            avg_mdd = np.mean(self.performance_stats["mdds"])
            avg_tpy = float(np.mean(self.performance_stats["annual_trades"]))
            logger.info(f"[QA] 성능: 평균 수익 {avg_ret:.2f}% | 평균 MDD {avg_mdd:.2f}% | 연간 거래 {avg_tpy:.1f}")

        # 3. [V14] Self-Healing Status Determination (Numerical Triggers)
        # =============================================================
        status = "SOFT" # Default
        
        # A. RIGID Check (Exploration Stagnation)
        is_rigid = (rej_rate > 0.90) or (pass_rate < 0.01)
        
        # B. COLLAPSED Check (Search Space Narrowing)
        coll_rate = self.similarity_stats.get("collision_rate", 0.0)
        avg_jaccard = self.similarity_stats.get("avg_jaccard", 0.0)
        # unique_ratio < 0.3 means collision_rate > 0.7
        is_collapsed = (avg_jaccard > 0.70) or (coll_rate > 0.70)
        
        if is_rigid:
            status = "RIGID"
        elif is_collapsed:
            status = "COLLAPSED"
            
        # Optional: Stage-specific health rules from config
        health_rules = getattr(config, 'STAGE_HEALTH_RULES', {})
        health_cfg = health_rules.get(self.stage_name, {})
        if "median_tpy_range" in health_cfg:
            t_min, t_max = health_cfg["median_tpy_range"]
            if avg_tpy > 0 and avg_tpy < t_min:
                status += "_LOW_ACTIVITY"
        
        logger.info(f"[QA] 상태: {status}")
        
        # [V14] Persist for Dashboard
        report_data = self._persist_diagnostics(
            pass_rate, 
            rej_rate, 
            status, 
            taxonomy if self.failure_counts else {},
            similarity=self.similarity_stats
        )

        logger.info("[QA] 진단 종료")
        
        # Add rich metrics for AutoTuner
        report_data["mean_tpy"] = avg_tpy
        report_data["zero_trade_ratio"] = self.performance_stats["zero_trades"] / self.total_samples if self.total_samples > 0 else 0
        report_data["best_score"] = max(self.performance_stats.get("scores", [config.EVAL_SCORE_MIN])) # We should collect scores
        
        return report_data

    def _persist_diagnostics(self, pass_rate, rej_rate, status, taxonomy, similarity=None):
        try:
            diag_dir = Path(getattr(config, 'LEDGER_DIR', 'ledger')) / "diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            
            # Group by stage
            stage_file = diag_dir / f"latest_{self.stage_name}.json"
            
            report_data = {
                "timestamp": time.time(),
                "stage": self.stage_name,
                "total_samples": self.total_samples,
                "pass_rate": pass_rate,
                "rejection_rate": rej_rate,
                "status": status,
                "taxonomy": taxonomy,
                "similarity": similarity or {},
                "top_reasons": dict(sorted(self.failure_counts.items(), key=lambda x: x[1], reverse=True)[:5])
            }
            
            with open(stage_file, "w") as f:
                json.dump(report_data, f, indent=2)
                
            # Also keep history (last 50 batches)
            history_file = diag_dir / "history.jsonl"
            with open(history_file, "a") as f:
                f.write(json.dumps(report_data) + "\n")
            
            return report_data
                
        except Exception as e:
            logger.error(f"[QA] 진단 저장 실패: {e}")
            return {"status": status, "pass_rate": pass_rate}


# Dataclasses moved to src.evaluation.contracts


def _split_contiguous_windows(df: pd.DataFrame, count: int, prefix: str) -> List[Tuple[str, pd.DataFrame]]:
    if df.empty:
        return []
    count = max(1, count)
    total = len(df)
    window_size = max(config.EVAL_MIN_WINDOW_BARS, total // count)
    windows = []
    for i in range(count):
        start = i * window_size
        end = total if i == count - 1 else min(total, start + window_size)
        if end - start < config.EVAL_MIN_WINDOW_BARS:
            break
        win_id = f"{prefix}_{i}"
        windows.append((win_id, df.iloc[start:end]))
    return windows


def _sample_reduced_windows(df: pd.DataFrame, slices: int, slice_bars: int) -> List[Tuple[str, pd.DataFrame]]:
    if df.empty:
        return []
    total = len(df)
    slice_len = min(slice_bars, total)
    if slice_len < config.EVAL_MIN_WINDOW_BARS:
        return _split_contiguous_windows(df, 1, "REDUCED")
    max_start = max(0, total - slice_len)
    step = max(1, max_start // max(1, slices - 1))
    windows = []
    for i in range(slices):
        start = min(max_start, i * step)
        end = start + slice_len
        win_id = f"REDUCED_{i}"
        windows.append((win_id, df.iloc[start:end]))
    return windows


def _get_eval_caps(stage_id: int) -> Dict[str, int]:
    caps = getattr(config, "EVAL_CAPS_BY_STAGE", {}) or {}
    stage_caps = caps.get(int(stage_id), {}) or {}
    return {
        "max_windows": int(stage_caps.get("max_windows", 0) or 0),
        "max_risk_samples": int(stage_caps.get("max_risk_samples", 0) or 0),
        "max_wf_splits": int(stage_caps.get("max_wf_splits", 0) or 0),
    }


def build_windows(df: pd.DataFrame, stage: str, stage_id: Optional[int] = None) -> List[Tuple[str, pd.DataFrame]]:
    current_stage_idx = int(stage_id or getattr(config, "CURRICULUM_CURRENT_STAGE", 1))
    caps = _get_eval_caps(current_stage_idx)
    
    if stage == "fast":
        lookback = min(config.EVAL_FAST_LOOKBACK_BARS, len(df))
        fast_df = df.iloc[-lookback:]
        cnt = config.EVAL_WINDOW_COUNT_FAST
        if caps["max_windows"] > 0:
            cnt = min(cnt, caps["max_windows"])
        return _split_contiguous_windows(fast_df, cnt, "FAST")
    
    if stage == "reduced":
        slices = config.EVAL_REDUCED_SLICES
        if caps["max_windows"] > 0:
            slices = min(slices, caps["max_windows"])
        return _sample_reduced_windows(df, slices, config.EVAL_REDUCED_SLICE_BARS)
    
    # [V11.3] Full Evaluation - Dynamic WF Splits
    if config.WF_GATE_ENABLED:
        stage_cfg = config.CURRICULUM_STAGES.get(current_stage_idx)
        if stage_cfg and getattr(stage_cfg, "wf_splits", None):
            cnt = int(stage_cfg.wf_splits)
        else:
            if current_stage_idx == 1:
                cnt = config.WF_SPLITS_STAGE1
            elif current_stage_idx == 2:
                cnt = config.WF_SPLITS_STAGE2
            else:
                cnt = config.WF_SPLITS_STAGE3
        if caps["max_wf_splits"] > 0:
            cnt = min(cnt, caps["max_wf_splits"])
    else:
        cnt = config.EVAL_WINDOW_COUNT_FULL

    if caps["max_windows"] > 0:
        cnt = min(cnt, caps["max_windows"])
        
    return _split_contiguous_windows(df, cnt, "FULL")


def _build_sample_risk_budget(
    base_risk: Dict[str, object],
    tp_pct: float,
    sl_pct: float,
    horizon: int,
) -> Dict[str, object]:
    est_vol = config.RISK_EST_DAILY_VOL
    k_up = tp_pct / est_vol if est_vol > 0 else 1.0
    k_down = sl_pct / est_vol if est_vol > 0 else 1.0
    risk_reward_ratio = (tp_pct / sl_pct) if sl_pct > 0 else 1.0

    risk_budget = dict(base_risk or {})
    risk_budget.update({
        "k_up": round(k_up, 2),
        "k_down": round(k_down, 2),
        "horizon": int(horizon),
        "stop_loss": round(sl_pct, 6),
        "risk_reward_ratio": round(risk_reward_ratio, 3),
        "tp_pct": round(tp_pct, 6),
        "sl_pct": round(sl_pct, 6),
    })
    return risk_budget


def _get_cached_signals(
    policy_spec: PolicySpec,
    features_df: pd.DataFrame,
    policy_sig: str,
    signal_cache: object,
) -> Tuple[pd.Series, pd.Series, float]:
    logic_sig = calculate_sha256({
        "policy_sig": policy_sig,
        "logic_trees": policy_spec.logic_trees,
        "decision_rules": policy_spec.decision_rules,
    })
    cache_key = signal_cache.make_key(features_df, {"logic_sig": logic_sig})
    cached = signal_cache.get(cache_key)
    if cached is not None:
        return cached

    from src.orchestration.policy_evaluator import RuleEvaluator
    evaluator = RuleEvaluator()
    entry_sig, exit_sig, complexity = evaluator.evaluate_signals(features_df, policy_spec)
    signal_cache.set(cache_key, (entry_sig, exit_sig, complexity))
    return entry_sig, exit_sig, complexity


def _evaluate_policy_windows(
    policy_spec: PolicySpec,
    df: pd.DataFrame,
    stage: str,
    regime_id: str,
    sample_count: int,
    window_sig: str,
    eval_sig: str,
    dataset_sig: str,
    stage_id: Optional[int] = None,
    pre_X: Optional[pd.DataFrame] = None, # [V12-O] Pre-calculated features
) -> EvaluationResult:
    try:
        stage_id = int(stage_id or getattr(config, "CURRICULUM_CURRENT_STAGE", 1))
        config.CURRICULUM_CURRENT_STAGE = stage_id

        if df.empty:
            return ModuleResult(policy_spec=policy_spec, module_key="", score=config.EVAL_SCORE_MIN, window_results=[], best_sample=None, stage=stage)

        df_local = df.copy()
        df_local.columns = [c.lower() for c in df_local.columns]
        
        # [V12-O] Use pre-calculated X if available
        if pre_X is not None:
            X_full = pre_X
        else:
            X_full = _generate_features_cached(
                df_values=df_local.values,
                df_columns=df_local.columns.tolist(),
                df_index=df_local.index,
                genome=policy_spec.feature_genome,
            )

        risk_budget = policy_spec.risk_budget or {}
        risk_profile_id = risk_budget.get("risk_profile", "DEFAULT")
        base_sl = risk_budget.get("stop_loss")
        base_rr = risk_budget.get("risk_reward_ratio")
        base_tp = None
        if base_sl is not None and base_rr is not None:
            try:
                base_tp = float(base_sl) * float(base_rr)
            except (TypeError, ValueError):
                base_tp = None
        base_h = risk_budget.get("horizon")

        tp_dist, sl_dist, h_dist = get_risk_distributions(
            template_id=policy_spec.template_id,
            regime_id=regime_id,
            risk_profile_id=risk_profile_id,
            base_tp=base_tp,
            base_sl=base_sl,
            base_h=base_h,
        )

        entry_threshold_id = f"th{config.EVAL_ENTRY_THRESHOLD:.2f}_mp{config.EVAL_ENTRY_MAX_PROB:.2f}"
        data_window_id = str(policy_spec.data_window.get("lookback", "full"))
        cost_bps = float(policy_spec.execution_assumption.get("cost_bps", 5))
        cost_model_id = f"bps{int(round(cost_bps))}"

        module_key = build_module_key(
            template_id=policy_spec.template_id,
            regime_id=regime_id,
            tp_dist_id=tp_dist.dist_id,
            sl_dist_id=sl_dist.dist_id,
            horizon_dist_id=h_dist.dist_id,
            entry_threshold_id=entry_threshold_id,
            data_window_id=data_window_id,
            cost_model_id=cost_model_id,
        )

        windows = build_windows(df_local, stage, stage_id=stage_id)
        window_results: List[WindowResult] = []
        best_sample: Optional[BestSample] = None
        policy_sig = get_policy_sig(policy_spec)
        signal_cache = get_signal_cache()
        backtest_cache = get_backtest_cache()

        for window_id, window_df in windows:
            cache_key = f"{policy_sig}:{window_sig}:{window_id}"
            cached_window = backtest_cache.get(cache_key)
            if cached_window:
                window_results.append(cached_window["window_result"])
                cached_best = cached_window.get("best_sample")
                if cached_best and (best_sample is None or cached_best.sample_score > best_sample.sample_score):
                    best_sample = cached_best
                continue

            X_window = X_full.loc[window_df.index]
            entry_sig, exit_sig, complexity = _get_cached_signals(
                policy_spec=policy_spec,
                features_df=X_window,
                policy_sig=policy_sig,
                signal_cache=signal_cache,
            )
            if complexity < 0:
                err_info = getattr(policy_spec, "_logictree_error", {}) or {}
                return EvaluationResult(
                    policy_spec=policy_spec,
                    module_key=getattr(config, "LOGICTREE_FAIL_ACTION", "INVALID_SPEC"),
                    score=config.EVAL_SCORE_MIN,
                    window_results=[],
                    best_sample=None,
                    stage=stage,
                    metadata={"error": err_info, "eval_sig": eval_sig, "dataset_sig": dataset_sig}
                )

            seed = stable_hash(f"{module_key}|{window_id}")
            samples = sample_risk_params(tp_dist, sl_dist, h_dist, sample_count, seed)

            sample_scores: List[float] = []
            violations: List[bool] = []
            alphas: List[float] = []
            window_best_sample: Optional[BestSample] = None

            for sample in samples:
                sample_risk = _build_sample_risk_budget(risk_budget, sample.tp_pct, sample.sl_pct, sample.horizon)
                
                # [Optimization] Re-use base_signals, only re-run physical backtest
                evaluator = RealEvaluator(cost_bps=cost_bps)
                metrics, bt = evaluator.evaluate(
                    df=window_df,
                    entry_signals=entry_sig,
                    exit_signals=exit_sig,
                    risk_budget=sample_risk,
                    target_regime=regime_id,
                    complexity_score=complexity
                )

                # Alpha = Strat ROI - Bench ROI
                alpha = evaluator.get_alpha(metrics)
                alphas.append(alpha)

                # [vNext] Validation (Hard Gate)
                passed, reason = validate_sample(metrics)
                failure_codes = [f.code for f in collect_validation_failures(metrics)]
                
                # [V14] Track rejection for QA
                from dataclasses import replace
                metrics = replace(
                    metrics,
                    is_rejected=(not passed),
                    rejection_reason=reason,
                    failure_codes=failure_codes,
                )

                if not passed:
                    sample_score = float(config.EVAL_SCORE_MIN)
                    violation = True
                else:
                    sample_score = score_sample(metrics)
                    violation = False

                sample_scores.append(sample_score)
                violations.append(violation)
                
                # [V18] Event Recording
                record_event("SAMPLE_EVALUATED", policy_id=policy_spec.spec_id, stage=stage, payload={"passed": not violation, "score": sample_score})
                
                if window_best_sample is None or sample_score > window_best_sample.sample_score:
                    # Capture core artifacts only for the best sample to save memory
                    core = _run_experiment_core(
                        policy_spec, window_df, X_window, sample_risk, include_trade_logic=False
                    )
                    window_best_sample = BestSample(
                        sample_id=sample.sample_id,
                        window_id=window_id,
                        risk_budget=sample_risk,
                        metrics=metrics,
                        sample_score=sample_score,
                        core=core,
                        X_features=None, # [V14-O] Slimmed
                        window_df=None # [V14-O] Slimmed
                    )

            agg = aggregate_sample_scores(sample_scores, violations)
            avg_alpha = float(np.mean(alphas)) if alphas else 0.0
            
            if window_best_sample and (best_sample is None or window_best_sample.sample_score > best_sample.sample_score):
                best_sample = window_best_sample

            window_result = WindowResult(
                window_id=window_id,
                raw_score=agg["score"],
                median_score=agg["median_score"],
                p10_score=agg["p10_score"],
                std_score=agg["std_score"],
                violation_rate=agg["violation_rate"],
                avg_alpha=avg_alpha,
                complexity_score=metrics.complexity_score
            )
            window_results.append(window_result)
            backtest_cache.set(cache_key, {"window_result": window_result, "best_sample": window_best_sample})

            # [V14-O] Early Exit: If this window failed hard gates significantly, skip remaining windows
            if agg["violation_rate"] > 0.8:
                record_event("WINDOW_EARLY_EXIT", policy_id=policy_spec.spec_id, stage=stage, payload={"window_id": window_id})
                logger.debug(f"[조기종료] {policy_spec.template_id} -> {window_id}에서 탈락")
                break

        # [V11.4] After evaluation, compute trade logic for the overall best sample to provide feedback
        if best_sample and "trade_logic" not in best_sample.core:
            # We need X_features for that window
            best_window_df = next(win_df for win_id, win_df in windows if win_id == best_sample.window_id)
            X_best = X_full.loc[best_window_df.index]
            
            best_core_with_logic = _run_experiment_core(
                policy_spec=policy_spec,
                df=best_window_df,
                X_features=X_best,
                risk_budget=best_sample.risk_budget,
                include_trade_logic=True,
            )
            best_sample.core["trade_logic"] = best_core_with_logic.get("trade_logic", {})
            
        # [V16] Reward Breakdown calculation for learning SSOT
        reward_breakdown = None
        if best_sample:
            from src.shared.metrics import aggregate_windows, metrics_to_legacy_dict
            legacy_metrics = metrics_to_legacy_dict(
                aggregate_windows([best_sample.metrics], eval_score_override=config.EVAL_SCORE_MIN)
            )
            shaper = get_reward_shaper()
            reward_breakdown = asdict(shaper.compute_breakdown(legacy_metrics))

        return EvaluationResult(
            policy_spec=policy_spec,
            module_key=module_key,
            score=config.EVAL_SCORE_MIN,
            window_results=window_results,
            best_sample=best_sample,
            stage=stage,
            reward_breakdown=reward_breakdown,
            fingerprint=generate_policy_id(policy_spec),
            metadata={"eval_sig": eval_sig, "dataset_sig": dataset_sig}
        )
    except Exception as e:
        import traceback
        exc_type = type(e).__name__
        err_msg = f"Worker Exception ({exc_type}): {str(e)}\n{traceback.format_exc()}"
        logger.error(f"[{stage}] 평가 실패: {policy_spec.template_id} ({err_msg})")
        return EvaluationResult(
            policy_spec=policy_spec,
            module_key="ERROR",
            score=config.EVAL_SCORE_MIN,
            window_results=[],
            best_sample=None,
            stage=stage,
            metadata={"error": err_msg, "exc_type": exc_type}
        )


def _normalize_window_scores(results: List[ModuleResult]) -> None:
    window_ids = sorted({w.window_id for r in results for w in r.window_results})
    for window_id in window_ids:
        scores = [w.raw_score for r in results for w in r.window_results if w.window_id == window_id]
        norm_scores = normalize_scores(scores)
        idx = 0
        for r in results:
            for w in r.window_results:
                if w.window_id == window_id:
                    w.normalized_score = norm_scores[idx]
                    idx += 1


def _finalize_module_scores(results: List[ModuleResult]) -> None:
    current_stage_idx = getattr(config, 'CURRICULUM_CURRENT_STAGE', 1)
    
    for r in results:
        if not r.window_results:
            r.score = config.EVAL_SCORE_MIN
            continue
            
        norm_scores = [w.normalized_score for w in r.window_results]
        arr = np.array(norm_scores, dtype=float)
        median_score = float(np.median(arr))
        p10_score = float(np.quantile(arr, config.EVAL_LOWER_QUANTILE))
        std_score = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        violation_rate = float(np.mean([w.violation_rate for w in r.window_results]))

        # [V15] Stage-based Walk-forward Alpha Consistency (Soft/Hard)
        stage_id = config.CURRICULUM_CURRENT_STAGE
        spec = config.CURRICULUM_STAGES.get(stage_id)
        
        alpha_floor = spec.alpha_floor if spec else -5.0
        gate_mode = spec.wf_gate_mode if spec else "soft"
        
        alphas = [w.avg_alpha for w in r.window_results]
        consistent_alpha = all(a >= alpha_floor for a in alphas)
        
        if not consistent_alpha and gate_mode == "hard":
            r.score = config.EVAL_SCORE_MIN
            continue
            
        # Base Score Calc
        final_score = (
            median_score
            + (config.EVAL_SCORE_W_RETURN_Q * p10_score)
            - (config.EVAL_SCORE_W_VOL * std_score)
            - (config.EVAL_SCORE_W_VIOLATION * violation_rate)
        )
        
        # Apply Soft Penalty if not consistent
        if not consistent_alpha:
            worst_alpha = min(alphas)
            penalty = abs(worst_alpha - alpha_floor) * 10.0
            final_score -= penalty
            
        r.score = max(float(config.EVAL_SCORE_MIN), float(final_score))


def evaluate_stage(
    policies: List[PolicySpec],
    df: pd.DataFrame,
    stage: str,
    regime_id: str,
    n_jobs: int,
    stage_id: Optional[int] = None,
    feature_map: Optional[Dict[str, pd.DataFrame]] = None, # [V12-O] Optional map
) -> Tuple[List[EvaluationResult], dict]:
    if not policies:
        return [], {"status": "EMPTY"}

    stage_id = int(stage_id or getattr(config, "CURRICULUM_CURRENT_STAGE", 1))
    config.CURRICULUM_CURRENT_STAGE = stage_id
    eval_caps = _get_eval_caps(stage_id)

    if stage == "fast":
        sample_count = config.EVAL_RISK_SAMPLES_FAST
    elif stage == "reduced":
        sample_count = config.EVAL_RISK_SAMPLES_REDUCED
    else:
        # [V12-O] Curriculum-aware sample count
        stage_cfg = config.CURRICULUM_STAGES.get(stage_id)
        if stage_cfg is not None and hasattr(stage_cfg, "eval_samples"):
            sample_count = getattr(stage_cfg, "eval_samples")
        else:
            sample_count = config.EVAL_RISK_SAMPLES_FULL

    requested_samples = sample_count
    if eval_caps["max_risk_samples"] > 0:
        sample_count = min(sample_count, eval_caps["max_risk_samples"])

    # [V15] Check ResultStore first
    store = get_result_store()
    dataset_sig = hash_dataframe(df)
    eval_sig = get_eval_config_signature()
    window_sig = f"{stage}_{regime_id}_s{stage_id}_{dataset_sig}_{eval_sig}"
    remaining_policies = []
    final_results = [None] * len(policies)
    window_count = len(build_windows(df, stage, stage_id=stage_id))
    
    # Use any available window signature logic. Here we use 'stage' as part of sig.
    for i, p in enumerate(policies):
        p_sig = get_policy_sig(p)
        w_sig = window_sig
        cached = store.get(p_sig, w_sig)
        if cached:
            # logger.debug(f"[ResultStore] Hit for {p.template_id}")
            final_results[i] = cached
        else:
            remaining_policies.append((i, p))
            
    if remaining_policies:
        indices, subset = zip(*remaining_policies)
        logger.info(f"[평가] ResultStore 필터 후 {len(subset)}/{len(policies)}개 실행")
        
        from src.shared.logger import _log_queue, setup_worker_logging
        from src.orchestration.parallel_manager import get_parallel_pool
        def worker_setup(q):
            setup_worker_logging(q)

        chunk_size = getattr(config, "PARALLEL_CHUNK_SIZE", 10)
        
        # [V14-O] Chunking Stage 2
        def process_chunk_s2(chunk_subset: List[PolicySpec], df_v, df_c, df_i, stage_name, r_id, samples, w_sig, e_sig, d_sig, stage_value):
            # Reconstruct DataFrame locally to avoid serialization of the object itself
            df_local = pd.DataFrame(df_v, columns=df_c, index=df_i).copy()
            config.CURRICULUM_CURRENT_STAGE = int(stage_value)
            results_chunk = []
            for p in chunk_subset:
                res = _evaluate_policy_windows(
                    p, df_local, stage_name, r_id, samples, w_sig, e_sig, d_sig, stage_id=stage_value
                )
                results_chunk.append(res)
            return results_chunk

        # Prepare numpy data
        df_v = df.values
        df_c = df.columns.tolist()
        df_i = df.index
        
        chunks = [subset[i:i + chunk_size] for i in range(0, len(subset), chunk_size)]

        s2_start = time.time()
        chunked_results = get_parallel_pool()(
            delayed(process_chunk_s2)(c, df_v, df_c, df_i, stage, regime_id, sample_count, window_sig, eval_sig, dataset_sig, stage_id)
            for c in chunks
        )
        s2_duration = time.time() - s2_start
        
        # Flatten
        subset_results = [res for chunk in chunked_results for res in chunk]
        
        # Record S2
        p_rate = len([r for r in subset_results if r.module_key != "ERROR"]) / len(subset_results) if subset_results else 0
        get_instrumentation().record_stage2(s2_duration, p_rate)
        
        for i, res in zip(indices, subset_results):
            p_sig = get_policy_sig(policies[i])
            w_sig = window_sig
            store.put(p_sig, w_sig, res)
            final_results[i] = res

    results = final_results

    # [V14] Aggregate QA across all modules
    qa_collector = OperationalQACollector(stage)
    qa_collector.similarity_analysis(policies) # [V14] Check batch redundancy early
    for res in results:
        if res.best_sample:
            qa_collector.collect(
                passed=not res.best_sample.metrics.is_rejected,
                reason=res.best_sample.metrics.rejection_reason,
                metrics=res.best_sample.metrics
            )
        elif res.module_key == "ERROR":
            qa_collector.collect(
                passed=False,
                reason=f"EXCEPTION_{res.metadata.get('exc_type', 'UNKNOWN')}",
                metrics=None
            )
    diagnostic_status = qa_collector.report() or {"status": "EMPTY"}
    diagnostic_status["eval_usage"] = {
        "stage_id": stage_id,
        "stage_label": stage,
        "window_count": window_count,
        "sample_count": sample_count,
        "sample_count_requested": requested_samples,
        "max_windows_cap": eval_caps["max_windows"],
        "max_risk_samples_cap": eval_caps["max_risk_samples"],
        "max_wf_splits_cap": eval_caps["max_wf_splits"],
    }

    _normalize_window_scores(results)
    _finalize_module_scores(results)

    # Episode-level observability (Cycle SSOT)
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
        )
        log_episode(summary)

    return results, diagnostic_status


def evaluate_v12_batch(
    policies: List[PolicySpec],
    df: pd.DataFrame,
    regime_id: str,
    n_jobs: int,
    stage_id: Optional[int] = None,
) -> Tuple[List[EvaluationResult], dict]:
    """
    V12-BT Batch Evaluation Orchestrator
    [V14-O] Optimized with chunking, pool reuse, and reduced data transfer.
    """
    if not policies:
        return [], {"status": "EMPTY"}

    stage_id = int(stage_id or getattr(config, "CURRICULUM_CURRENT_STAGE", 1))
    config.CURRICULUM_CURRENT_STAGE = stage_id

    # 0. Prepare Data for Workers (numpy-only to avoid massive serialization)
    df_values = df.values
    df_columns = df.columns.tolist()
    df_index = df.index
    
    # [V14-O] Load settings
    from src.orchestration.parallel_manager import get_parallel_pool
    chunk_size = getattr(config, "PARALLEL_CHUNK_SIZE", 10)
    inst = get_instrumentation()

    # 1. Stage 1: Fast Filter (All Policies)
    logger.info(f"[평가] 1단계(빠른 필터): 후보 {len(policies)}개 (청크 {chunk_size})")
    s1_start = time.time()
    
    # [V14-O] Chunking Stage 1 to reduce IPC overhead
    def process_chunk_s1(chunk: List[PolicySpec], values, columns, index, stage_value):
        from src.orchestration.policy_evaluator import RuleEvaluator
        config.CURRICULUM_CURRENT_STAGE = int(stage_value)
        evaluator = RuleEvaluator()
        signals = []
        for p in chunk:
            X = _generate_features_cached(values, columns, index, p.feature_genome)
            # Stage 1 Optimization: Skip backtest engine, just get entry signal
            entry_sig, _, _ = evaluator.evaluate_signals(X, p)
            signals.append(entry_sig)
        return signals

    # Split into chunks
    chunks = [policies[i:i + chunk_size] for i in range(0, len(policies), chunk_size)]
    
    # Run Stage 1 (Single pool call)
    chunked_signals = get_parallel_pool()(
        delayed(process_chunk_s1)(c, df_values, df_columns, df_index, stage_id) 
        for c in chunks
    )
    
    # Flatten signals
    all_signals = [sig for chunk in chunked_signals for sig in chunk]
    
    fast_filter = FastFilter()
    fast_scores = fast_filter.score_batch(df, all_signals, [p.risk_budget for p in policies])
    s1_duration = time.time() - s1_start
    
    # Record S1 Metrics
    s1_pass_count = len([s for s in fast_scores if s > config.EVAL_SCORE_MIN])
    inst.record_stage1(s1_duration, s1_pass_count / len(policies) if policies else 0)
    
    scored_policies = list(zip(fast_scores, policies))
    
    # Selection (Mixed Strategy: Elites + Explorers)
    top_n = getattr(config, "V12_ELITE_TOP_N", 5)
    stage_cfg = config.CURRICULUM_STAGES.get(stage_id, {})
    exploration_slot = float(getattr(stage_cfg, "exploration_slot", 0.2))
    
    from src.evaluation.dual_stage import DualStageEvaluator
    ds_eval = DualStageEvaluator()
    promote_total = max(top_n, int(round(len(scored_policies) * config.EVAL_FAST_TOP_PCT)))
    promote_total = min(promote_total, len(scored_policies))
    explorer_n = int(round(promote_total * exploration_slot)) if promote_total > 0 else 0
    if exploration_slot > 0 and promote_total > 1 and explorer_n == 0:
        explorer_n = 1
    elite_n = max(1, promote_total - explorer_n) if promote_total > 0 else 0
    promoted_policies = ds_eval.select_mixed_candidates(
        scored_policies,
        elite_n=elite_n,
        explorer_n=explorer_n
    )
    promoted_templates = {p.template_id for p in promoted_policies}
    
    cold_ratio = explorer_n / max(1, promote_total) if promote_total > 0 else 0.0
    if hasattr(inst, "record_exploration"):
        inst.record_exploration(cold_ratio)
    logger.info(f"[평가] 승격 믹스: 엘리트 {elite_n} | 콜드스타트 {explorer_n} ({cold_ratio:.1%})")
    logger.info(f"[평가] 2단계(정밀 평가): 승격 {len(promoted_policies)}개")
    
    # 2. Stage 2: Detailed Evaluation (Promoted only)
    # feature_map=None as it will be generated locally in workers
    promoted_results, diagnostic_status = evaluate_stage(
        promoted_policies,
        df,
        "full",
        regime_id,
        n_jobs,
        stage_id=stage_id,
        feature_map=None,
    )
    
    # 3. Handle Filtered (Give them base scores/ModuleResults)
    promoted_map = {res.policy_spec.template_id: res for res in promoted_results}
    final_results = []
    
    for i, p in enumerate(policies):
        if p.template_id in promoted_map:
            final_results.append(promoted_map[p.template_id])
        else:
            # Create a placeholder ModuleResult for others
            # Mark score as EVAL_SCORE_MIN but we can add feedback later if needed
            final_results.append(EvaluationResult(
                policy_spec=p,
                module_key="filtered_out",
                score=config.EVAL_SCORE_MIN,
                window_results=[],
                best_sample=None,
                stage="filtered",
                fingerprint=generate_policy_id(p)
            ))
            
    return final_results, diagnostic_status


def select_top_modules(results: List[EvaluationResult], top_pct: float, top_k: int = 0) -> List[EvaluationResult]:
    if not results:
        return []
    results_sorted = sorted(results, key=lambda r: r.score, reverse=True)
    if top_k > 0:
        return results_sorted[:min(top_k, len(results_sorted))]
    keep_n = max(1, int(round(len(results_sorted) * top_pct)))
    return results_sorted[:keep_n]


def persist_best_samples(
    repo,
    results: List[EvaluationResult],
    df: pd.DataFrame, # [V14-O] Added df for reconstruction
    existing_ids: Optional[set] = None,
) -> int:
    """
    [V16] One-Pass Persistence
    Saves already evaluated results to the ledger without re-calculating.
    [V14-O] Optimized: Reconstructs window_df and X_features from df to save IPC transfer.
    """
    saved = 0
    skipped = 0
    existing_ids = existing_ids or set()
    
    # Pre-build windows for reconstruction
    # We assume stage "full" or whatever the results were evaluated at.
    # Results should have the stage info. 
    # Since persist usually happens after full eval, we use results[0].stage
    if not results:
        return 0
    
    stage = results[0].stage
    stage_id = int(getattr(config, "CURRICULUM_CURRENT_STAGE", 1))
    windows = build_windows(df, stage, stage_id=stage_id)
    window_map = {win_id: win_df for win_id, win_df in windows}
    df_values = df.values
    df_columns = df.columns.tolist()
    df_index = df.index

    for r in results:
        # Duplication check (Structural ID)
        if r.policy_spec.spec_id in existing_ids:
            skipped += 1
            continue

        best = r.best_sample
        
        # [V18] Handle results without best_sample (e.g., Stage 1 filtered out)
        if best is None:
            # Create minimal scorecard for rejected record
            scorecard = {
                "eval_score": r.score,
                "eval_stage": r.stage,
                "module_key": r.module_key,
                "fingerprint": r.fingerprint,
                "is_filtered": True
            }
            record, artifact = build_record_and_artifact(
                policy_spec=r.policy_spec,
                df=df.iloc[:5], # Dummy small DF
                X_features=pd.DataFrame(),
                core={}, # Empty core
                scorecard=scorecard,
                module_key=r.module_key,
                eval_stage=r.stage,
            )
            repo.save_record(record, artifact)
            saved += 1
            continue
            
        # [V16] Use pre-captured artifacts for candidates that have evaluation data
        spec = copy.deepcopy(r.policy_spec)
        spec.risk_budget = best.risk_budget
        
        # [V14-O] Reconstruct artifacts locally
        df_local = window_map.get(best.window_id)
        if df_local is None:
            # Fallback for FAST or other windows
            lookback = min(config.EVAL_FAST_LOOKBACK_BARS, len(df))
            df_local = df.iloc[-lookback:]
            
        # Re-generate features for the window
        X_full = _generate_features_cached(df_local.values, df_local.columns.tolist(), df_local.index, spec.feature_genome)
        
        scorecard = {
            "eval_score": r.score,
            "eval_stage": r.stage,
            "module_key": r.module_key,
            "sample_id": best.sample_id,
            "sample_window_id": best.window_id,
            
            # Standard keys (for validation & SSOT)
            "total_return_pct": best.metrics.equity.total_return_pct,
            "mdd_pct": best.metrics.equity.max_drawdown_pct,
            "sharpe": best.metrics.equity.sharpe,
            "win_rate": best.metrics.trades.win_rate,
            "n_trades": best.metrics.trades.trade_count,
            "reward_risk": best.metrics.trades.reward_risk,
            
            # Prefixed keys (for Dashboard API compatibility)
            "sample_total_return_pct": best.metrics.equity.total_return_pct,
            "sample_mdd_pct": best.metrics.equity.max_drawdown_pct,
            "sample_sharpe": best.metrics.equity.sharpe,
            "sample_excess_return": best.metrics.equity.excess_return,
            "sample_rr": best.metrics.trades.reward_risk,
            "sample_vol_pct": best.metrics.equity.vol_pct,
            "sample_trades": best.metrics.trades.trade_count,
            "sample_win_rate": best.metrics.trades.win_rate,
            
            "fingerprint": r.fingerprint,
            "reward_breakdown": r.reward_breakdown
        }

        record, artifact = build_record_and_artifact(
            policy_spec=spec,
            df=df_local,
            X_features=X_full,
            core=best.core,
            scorecard=scorecard,
            module_key=r.module_key,
            eval_stage=r.stage,
        )
        
        repo.save_record(record, artifact)
        saved += 1
    if saved > 0 or skipped > 0:
        logger.info(f"[저장] 평가 결과 저장: {saved} | 건너뜀 {skipped} (원패스)")
    return saved
