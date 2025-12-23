from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Dict, List, Optional, Tuple

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
)
from src.shared.backtest import run_signal_backtest
from src.orchestration.run_experiment import _generate_features_cached, _run_experiment_core, build_record_and_artifact
from src.shared.logger import get_logger

logger = get_logger("orchestration.evaluation")


@dataclass
class WindowResult:
    window_id: str
    raw_score: float
    median_score: float
    p10_score: float
    std_score: float
    violation_rate: float
    avg_alpha: float = 0.0  # [V11.3] Walk-forward Alpha Consistency
    normalized_score: float = 0.0


@dataclass
class BestSample:
    sample_id: str
    window_id: str
    risk_budget: Dict[str, float]
    metrics: SampleMetrics
    sample_score: float
    core: Dict[str, object]


@dataclass
class ModuleResult:
    policy_spec: PolicySpec
    module_key: str
    score: float
    window_results: List[WindowResult]
    best_sample: Optional[BestSample]
    stage: str


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


def build_windows(df: pd.DataFrame, stage: str) -> List[Tuple[str, pd.DataFrame]]:
    current_stage_idx = getattr(config, 'CURRICULUM_CURRENT_STAGE', 1)
    
    if stage == "fast":
        lookback = min(config.EVAL_FAST_LOOKBACK_BARS, len(df))
        fast_df = df.iloc[-lookback:]
        return _split_contiguous_windows(fast_df, config.EVAL_WINDOW_COUNT_FAST, "FAST")
    
    if stage == "reduced":
        return _sample_reduced_windows(df, config.EVAL_REDUCED_SLICES, config.EVAL_REDUCED_SLICE_BARS)
    
    # [V11.3] Full Evaluation - Dynamic WF Splits
    if config.WF_GATE_ENABLED:
        if current_stage_idx == 1:
            cnt = config.WF_SPLITS_STAGE1
        elif current_stage_idx == 2:
            cnt = config.WF_SPLITS_STAGE2
        else:
            cnt = config.WF_SPLITS_STAGE3
    else:
        cnt = config.EVAL_WINDOW_COUNT_FULL
        
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


def _evaluate_policy_windows(
    policy_spec: PolicySpec,
    df: pd.DataFrame,
    stage: str,
    regime_id: str,
    sample_count: int,
) -> ModuleResult:
    if df.empty:
        return ModuleResult(policy_spec=policy_spec, module_key="", score=config.EVAL_SCORE_MIN, window_results=[], best_sample=None, stage=stage)

    df_local = df.copy()
    df_local.columns = [c.lower() for c in df_local.columns]
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

    windows = build_windows(df_local, stage)
    window_results: List[WindowResult] = []
    best_sample: Optional[BestSample] = None

    for window_id, window_df in windows:
        X_window = X_full.loc[window_df.index]
        seed = stable_hash(f"{module_key}|{window_id}")
        samples = sample_risk_params(tp_dist, sl_dist, h_dist, sample_count, seed)

        sample_scores: List[float] = []
        violations: List[bool] = []
        alphas: List[float] = []

        for sample in samples:
            sample_risk = _build_sample_risk_budget(risk_budget, sample.tp_pct, sample.sl_pct, sample.horizon)
            core = _run_experiment_core(
                policy_spec=policy_spec,
                df=window_df,
                X_features=X_window,
                risk_budget=sample_risk,
                include_trade_logic=False,
            )
            bt = run_signal_backtest(
                price_df=window_df,
                results_df=core.get("results_df"),
                risk_budget=sample_risk,
                cost_bps=cost_bps,
                target_regime=regime_id, # [V11.2]
            )
            trade_returns = np.array([t["return_pct"] / 100.0 for t in bt.trades], dtype=float)
            metrics = compute_sample_metrics(trade_returns, bt.trade_count)
            metrics.total_return_pct = bt.total_return_pct
            metrics.mdd_pct = bt.mdd_pct
            metrics.win_rate = bt.win_rate
            metrics.trade_count = bt.trade_count
            metrics.valid_trade_count = bt.valid_trade_count # [V11.2]
            
            # [V11.2] Bench ROI (Buy & Hold) calculation for Alpha
            bench_start = float(window_df["close"].iloc[0])
            bench_end = float(window_df["close"].iloc[-1])
            bench_roi_pct = ((bench_end / bench_start) - 1.0) * 100.0
            metrics.benchmark_roi_pct = bench_roi_pct
            
            # Alpha = Strat ROI - Bench ROI
            alpha = (bt.total_return_pct - bench_roi_pct) / 100.0
            alphas.append(alpha)

            # [vNext] Calc Exposure Ratio
            results_df = core.get("results_df")
            if results_df is not None and not results_df.empty:
                exposure_ratio = float((results_df["pred"] != 0).sum()) / len(results_df)
            else:
                exposure_ratio = 0.0
            metrics.exposure_ratio = exposure_ratio

            # [vNext] Validation (Hard Gate)
            passed, reason = validate_sample(metrics)

            if not passed:
                sample_score = float(config.EVAL_SCORE_MIN)
                violation = True
            else:
                sample_score = score_sample(metrics)
                violation = False

            sample_scores.append(sample_score)
            violations.append(violation)

            if best_sample is None or sample_score > best_sample.sample_score:
                best_sample = BestSample(
                    sample_id=sample.sample_id,
                    window_id=window_id,
                    risk_budget=sample_risk,
                    metrics=metrics,
                    sample_score=sample_score,
                    core=core,
                )

        agg = aggregate_sample_scores(sample_scores, violations)
        avg_alpha = float(np.mean(alphas)) if alphas else 0.0
        
        window_results.append(
            WindowResult(
                window_id=window_id,
                raw_score=agg["score"],
                median_score=agg["median_score"],
                p10_score=agg["p10_score"],
                std_score=agg["std_score"],
                violation_rate=agg["violation_rate"],
                avg_alpha=avg_alpha,
            )
        )

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
        
    return ModuleResult(
        policy_spec=policy_spec,
        module_key=module_key,
        score=config.EVAL_SCORE_MIN,
        window_results=window_results,
        best_sample=best_sample,
        stage=stage,
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

        # [V11.3] Walk-forward Consistency Gate
        wf_failed = False
        if config.WF_GATE_ENABLED and r.stage == "full":
            if current_stage_idx == 1:
                alpha_floor = config.WF_ALPHA_FLOOR_STAGE1
            elif current_stage_idx == 2:
                alpha_floor = config.WF_ALPHA_FLOOR_STAGE2
            else:
                alpha_floor = config.WF_ALPHA_FLOOR_STAGE3
            
            # 모든 구간에서 Alpha 가 하한선 이상이어야 함 (또는 대다수)
            # 여기서는 '모근 구간 무조건 통과' 옵션 (Deterministic)
            for w in r.window_results:
                if w.avg_alpha < alpha_floor:
                    wf_failed = True
                    break

        if violation_rate > 0.5 or wf_failed:
             # [V11.3] WF 실패 시 Rejection 수준의 페널티
             r.score = config.EVAL_SCORE_MIN
        else:
            r.score = (
                median_score
                + (config.EVAL_SCORE_W_RETURN_Q * p10_score)
                - (config.EVAL_SCORE_W_VOL * std_score)
                - (config.EVAL_SCORE_W_VIOLATION * violation_rate)
            )
        
        # 하한선 적용 (RL 안정성)
        r.score = max(float(config.EVAL_SCORE_MIN), float(r.score))


def evaluate_stage(
    policies: List[PolicySpec],
    df: pd.DataFrame,
    stage: str,
    regime_id: str,
    n_jobs: int,
) -> List[ModuleResult]:
    if not policies:
        return []

    if stage == "fast":
        sample_count = config.EVAL_RISK_SAMPLES_FAST
    elif stage == "reduced":
        sample_count = config.EVAL_RISK_SAMPLES_REDUCED
    else:
        sample_count = config.EVAL_RISK_SAMPLES_FULL

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_evaluate_policy_windows)(policy, df, stage, regime_id, sample_count)
        for policy in policies
    )

    _normalize_window_scores(results)
    _finalize_module_scores(results)
    return results


def select_top_modules(results: List[ModuleResult], top_pct: float, top_k: int = 0) -> List[ModuleResult]:
    if not results:
        return []
    results_sorted = sorted(results, key=lambda r: r.score, reverse=True)
    if top_k > 0:
        return results_sorted[:min(top_k, len(results_sorted))]
    keep_n = max(1, int(round(len(results_sorted) * top_pct)))
    return results_sorted[:keep_n]


def persist_best_samples(
    repo,
    results: List[ModuleResult],
    df: pd.DataFrame,
) -> int:
    saved = 0
    for r in results:
        best = r.best_sample
        if best is None:
            continue
        spec = copy.deepcopy(r.policy_spec)
        spec.risk_budget = best.risk_budget
        windows = {win_id: win_df for win_id, win_df in build_windows(df, r.stage)}
        window_df = windows.get(best.window_id, df)
        df_local = window_df.copy()
        df_local.columns = [c.lower() for c in df_local.columns]
        X_full = _generate_features_cached(
            df_values=df_local.values,
            df_columns=df_local.columns.tolist(),
            df_index=df_local.index,
            genome=r.policy_spec.feature_genome,
        )
        scorecard = {
            "eval_score": r.score,
            "eval_stage": r.stage,
            "module_key": r.module_key,
            "sample_id": best.sample_id,
            "sample_window_id": best.window_id,
            "sample_total_return_pct": best.metrics.total_return_pct,
            "sample_mdd_pct": best.metrics.mdd_pct,
            "sample_rr": best.metrics.reward_risk,
            "sample_vol_pct": best.metrics.vol_pct,
            "sample_trades": best.metrics.trade_count,
            "sample_win_rate": best.metrics.win_rate,
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
    return saved
