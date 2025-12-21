from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import time

import numpy as np
import pandas as pd

from src.config import config
from src.contracts import LedgerRecord, PolicySpec
from src.l2_sl.artifacts import ArtifactBundle
from src.shared.returns import get_risk_unit
from src.shared.logger import get_logger
from src.orchestration.run_experiment import _generate_features_cached, _hash_payload, _run_experiment_core

logger = get_logger("orchestration.evaluator")


@dataclass
class EvalResult:
    policy_spec: PolicySpec
    score: float
    summary: Dict[str, Any]
    record: Optional[LedgerRecord] = None
    artifact: Optional[ArtifactBundle] = None


def _clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


def _derive_risk_budget(base_budget: Dict[str, Any], k_up: float, k_down: float, horizon: int) -> Dict[str, Any]:
    risk_budget = dict(base_budget or {})
    risk_profile = risk_budget.get("risk_profile", "DEFAULT")

    risk_reward_ratio = k_up / k_down if k_down > 0 else 1.0
    max_leverage = 0.5 + (1.0 / (k_down + 0.5))
    max_leverage = _clamp(max_leverage, 0.5, 1.5)

    stop_loss = k_down * 0.015
    stop_loss = _clamp(stop_loss, 0.005, 0.05)

    risk_budget.update({
        "k_up": round(k_up, 2),
        "k_down": round(k_down, 2),
        "horizon": int(horizon),
        "risk_profile": risk_profile,
        "stop_loss": round(stop_loss, 4),
        "max_leverage": round(max_leverage, 2),
        "risk_reward_ratio": round(risk_reward_ratio, 2),
    })
    return risk_budget


def sample_risk_budgets(
    base_budget: Dict[str, Any],
    count: int,
    jitter_pct: float,
    rng: np.random.RandomState,
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []

    base_k_up = base_budget.get("k_up")
    base_k_down = base_budget.get("k_down")
    base_horizon = base_budget.get("horizon")

    if base_k_up is None:
        base_k_up = (config.RISK_K_UP_MIN + config.RISK_K_UP_MAX) / 2.0
    if base_k_down is None:
        base_k_down = (config.RISK_K_DOWN_MIN + config.RISK_K_DOWN_MAX) / 2.0
    if base_horizon is None:
        base_horizon = int((config.RISK_HORIZON_MIN + config.RISK_HORIZON_MAX) / 2)

    budgets = [_derive_risk_budget(base_budget, base_k_up, base_k_down, base_horizon)]

    for _ in range(max(0, count - 1)):
        if jitter_pct > 0:
            k_up = base_k_up * (1 + rng.uniform(-jitter_pct, jitter_pct))
            k_down = base_k_down * (1 + rng.uniform(-jitter_pct, jitter_pct))
            h_low = max(config.RISK_HORIZON_MIN, int(base_horizon * (1 - jitter_pct)))
            h_high = min(config.RISK_HORIZON_MAX, int(base_horizon * (1 + jitter_pct)))
            horizon = rng.randint(h_low, max(h_low + 1, h_high + 1))
        else:
            k_up = base_k_up
            k_down = base_k_down
            horizon = base_horizon

        k_up = _clamp(k_up, config.RISK_K_UP_MIN, config.RISK_K_UP_MAX)
        k_down = _clamp(k_down, config.RISK_K_DOWN_MIN, config.RISK_K_DOWN_MAX)

        budgets.append(_derive_risk_budget(base_budget, k_up, k_down, horizon))

    return budgets


def split_time_windows(n_obs: int, window_count: int, min_window: int) -> List[Tuple[int, int]]:
    if n_obs <= 0:
        return []
    if window_count <= 1 or n_obs <= min_window:
        return [(0, n_obs)]

    max_windows = max(1, n_obs // max(1, min_window))
    window_count = min(window_count, max_windows)
    if window_count <= 1:
        return [(0, n_obs)]

    window_size = max(1, n_obs // window_count)
    windows = []
    for i in range(window_count):
        start = i * window_size
        end = n_obs if i == window_count - 1 else (i + 1) * window_size
        if end > start:
            windows.append((start, end))
    return windows or [(0, n_obs)]


def compute_trade_stats(pred: pd.Series, net_pnl: pd.Series) -> Dict[str, float]:
    if pred is None or pred.empty:
        return {"n_trades": 0.0, "win_rate": 0.0, "trade_score": 0.0}

    res_shift = pred.shift(1).fillna(0)
    is_trade_start = (pred != res_shift) & (pred != 0)
    n_trades = int(is_trade_start.sum())

    active_pnl = net_pnl[pred != 0]
    win_rate = float((active_pnl > 0).mean()) if len(active_pnl) > 0 else 0.0
    trade_score = min(n_trades / max(1, config.EVAL_TRADE_TARGET), 1.0)
    return {"n_trades": float(n_trades), "win_rate": win_rate, "trade_score": trade_score}


def compute_performance_metrics(net_pnl: pd.Series, risk_budget: Dict[str, Any]) -> Dict[str, float]:
    risk_unit = get_risk_unit(risk_budget, default_pct=0.01)
    returns = pd.to_numeric(net_pnl, errors="coerce").fillna(0.0)
    pct_returns = returns * risk_unit
    pct_returns = pct_returns.clip(lower=-0.95)

    if pct_returns.empty:
        return {
            "total_return_pct": 0.0,
            "mdd_pct": 0.0,
            "volatility_pct": 0.0,
            "reward_risk": 0.0,
            "stop_loss_pct": risk_unit * 100.0,
        }

    equity = (1.0 + pct_returns).cumprod()
    total_return_pct = float((equity.iloc[-1] - 1.0) * 100.0)

    drawdown = equity / equity.cummax() - 1.0
    mdd_pct = float(drawdown.min() * 100.0)

    vol_pct = float(pct_returns.std(ddof=1) * 100.0) if len(pct_returns) > 1 else 0.0

    gains = pct_returns[pct_returns > 0]
    losses = pct_returns[pct_returns < 0]
    avg_gain = float(gains.mean()) if len(gains) > 0 else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
    reward_risk = avg_gain / (avg_loss + 1e-9) if avg_loss > 0 else 0.0

    return {
        "total_return_pct": total_return_pct,
        "mdd_pct": mdd_pct,
        "volatility_pct": vol_pct,
        "reward_risk": reward_risk,
        "stop_loss_pct": risk_unit * 100.0,
    }


def summarize_eval_metrics(
    runs: List[Dict[str, Any]],
    risk_budgets: List[Dict[str, Any]],
    window_count: int,
    min_window: int,
    lower_quantile: float,
) -> Dict[str, float]:
    metric_rows: List[Dict[str, float]] = []

    for run, risk_budget in zip(runs, risk_budgets):
        results_df = run.get("results_df")
        if results_df is None or results_df.empty:
            continue
        windows = split_time_windows(len(results_df), window_count, min_window)
        for start, end in windows:
            window_df = results_df.iloc[start:end]
            if window_df.empty:
                continue
            perf = compute_performance_metrics(window_df["net_pnl"], risk_budget)
            trades = compute_trade_stats(window_df["pred"], window_df["net_pnl"])
            metric_rows.append({**perf, **trades})

    if not metric_rows:
        return {
            "eval_score": 0.0,
            "eval_return_median_pct": 0.0,
            "eval_return_pctl_pct": 0.0,
            "eval_return_mean_pct": 0.0,
            "eval_return_std_pct": 0.0,
            "eval_mdd_median_pct": 0.0,
            "eval_vol_median_pct": 0.0,
            "eval_rr_median": 0.0,
            "eval_trade_score_median": 0.0,
            "eval_norm_return_median": 0.0,
            "eval_norm_return_pctl": 0.0,
            "eval_norm_mdd_median": 0.0,
            "eval_norm_vol_median": 0.0,
            "eval_norm_return_mean": 0.0,
            "eval_norm_return_std": 0.0,
            "eval_risk_scale_pct": config.EVAL_RISK_SCALE_FLOOR,
        }

    metrics_df = pd.DataFrame(metric_rows)
    risk_scale = float(metrics_df["stop_loss_pct"].median()) if "stop_loss_pct" in metrics_df else 0.0
    risk_scale = max(risk_scale, config.EVAL_RISK_SCALE_FLOOR)

    metrics_df["norm_return"] = metrics_df["total_return_pct"] / risk_scale
    metrics_df["norm_mdd"] = metrics_df["mdd_pct"].abs() / risk_scale
    metrics_df["norm_vol"] = metrics_df["volatility_pct"] / risk_scale

    return_median = float(metrics_df["total_return_pct"].median())
    return_pctl = float(metrics_df["total_return_pct"].quantile(lower_quantile))
    return_mean = float(metrics_df["total_return_pct"].mean())
    return_std = float(metrics_df["total_return_pct"].std(ddof=1)) if len(metrics_df) > 1 else 0.0

    norm_return_median = float(metrics_df["norm_return"].median())
    norm_return_pctl = float(metrics_df["norm_return"].quantile(lower_quantile))
    norm_return_mean = float(metrics_df["norm_return"].mean())
    norm_return_std = float(metrics_df["norm_return"].std(ddof=1)) if len(metrics_df) > 1 else 0.0

    mdd_median = float(metrics_df["mdd_pct"].median())
    vol_median = float(metrics_df["volatility_pct"].median())
    rr_median = float(metrics_df["reward_risk"].median())
    trade_score_median = float(metrics_df["trade_score"].median())

    score = compute_eval_score(
        norm_return_median=norm_return_median,
        norm_return_pctl=norm_return_pctl,
        norm_mdd_median=float(metrics_df["norm_mdd"].median()),
        norm_vol_median=float(metrics_df["norm_vol"].median()),
        rr_median=rr_median,
        trade_score=trade_score_median,
        win_rate_median=float(metrics_df["win_rate"].median()),
    )

    return {
        "eval_score": score,
        "eval_return_median_pct": return_median,
        "eval_return_pctl_pct": return_pctl,
        "eval_return_mean_pct": return_mean,
        "eval_return_std_pct": return_std,
        "eval_mdd_median_pct": mdd_median,
        "eval_vol_median_pct": vol_median,
        "eval_rr_median": rr_median,
        "eval_trade_score_median": trade_score_median,
        "eval_win_rate_median": float(metrics_df["win_rate"].median()),
        "eval_norm_return_median": norm_return_median,
        "eval_norm_return_pctl": norm_return_pctl,
        "eval_norm_mdd_median": float(metrics_df["norm_mdd"].median()),
        "eval_norm_vol_median": float(metrics_df["norm_vol"].median()),
        "eval_norm_return_mean": norm_return_mean,
        "eval_norm_return_std": norm_return_std,
        "eval_risk_scale_pct": risk_scale,
    }


def compute_eval_score(
    norm_return_median: float,
    norm_return_pctl: float,
    norm_mdd_median: float,
    norm_vol_median: float,
    rr_median: float,
    trade_score: float,
    win_rate_median: float,
) -> float:
    return (
        (config.EVAL_SCORE_W_RETURN * norm_return_median)
        + (config.EVAL_SCORE_W_RETURN_Q * norm_return_pctl)
        - (config.EVAL_SCORE_W_MDD * norm_mdd_median)
        - (config.EVAL_SCORE_W_VOL * norm_vol_median)
        + (config.EVAL_SCORE_W_RR * rr_median)
        + (config.EVAL_SCORE_W_TRADE * trade_score)
        + (config.EVAL_SCORE_W_WINRATE * win_rate_median * 100.0)
    )


def select_recent_data(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if lookback <= 0 or len(df) <= lookback:
        return df
    return df.tail(lookback)


def select_reduced_data(df: pd.DataFrame, slice_bars: int, n_slices: int) -> pd.DataFrame:
    if slice_bars <= 0 or n_slices <= 1:
        return df

    total = len(df)
    if total <= slice_bars * n_slices:
        return df

    step = (total - slice_bars) // (n_slices - 1)
    slices = []
    for i in range(n_slices):
        start = i * step
        end = start + slice_bars
        slices.append(df.iloc[start:end])
    reduced = pd.concat(slices).sort_index()
    return reduced[~reduced.index.duplicated(keep="first")]


def evaluate_policy(
    policy_spec: PolicySpec,
    market_data: pd.DataFrame,
    risk_samples: int,
    window_count: int,
    min_window: int,
    include_trade_logic: bool = False,
    persist: bool = False,
) -> EvalResult:
    df = market_data.copy()
    df.columns = [c.lower() for c in df.columns]

    X_features = _generate_features_cached(
        df_values=df.values,
        df_columns=df.columns.tolist(),
        df_index=df.index,
        genome=policy_spec.feature_genome,
    )

    if X_features.empty:
        return EvalResult(policy_spec=policy_spec, score=0.0, summary={"eval_score": 0.0})

    seed_payload = f"{policy_spec.spec_id}-{len(df)}"
    rng = np.random.RandomState(abs(hash(seed_payload)) % (2**32))

    risk_budgets = sample_risk_budgets(
        base_budget=policy_spec.risk_budget or {},
        count=risk_samples,
        jitter_pct=config.EVAL_RISK_JITTER_PCT,
        rng=rng,
    )

    runs = []
    base_run = None
    for idx, risk_budget in enumerate(risk_budgets):
        run = _run_experiment_core(
            policy_spec=policy_spec,
            df=df,
            X_features=X_features,
            risk_budget=risk_budget,
            include_trade_logic=include_trade_logic and idx == 0,
        )
        runs.append(run)
        if idx == 0:
            base_run = run

    summary = summarize_eval_metrics(
        runs=runs,
        risk_budgets=risk_budgets,
        window_count=window_count,
        min_window=min_window,
        lower_quantile=config.EVAL_LOWER_QUANTILE,
    )
    summary["eval_risk_samples"] = len(risk_budgets)
    summary["eval_window_count"] = window_count

    score = float(summary.get("eval_score", 0.0))

    if not persist or base_run is None:
        return EvalResult(policy_spec=policy_spec, score=score, summary=summary)

    results_df = base_run.get("results_df")
    if results_df is None or results_df.empty:
        backtest_results = []
    else:
        backtest_results = results_df.reset_index().rename(columns={"index": "date"}).to_dict(orient="records")

    base_cpcv = base_run.get("cpcv", {}) or {}
    cpcv_metrics = dict(base_cpcv)
    cpcv_metrics.update(summary)

    hard_fail = []
    if base_cpcv.get("cpcv_worst", 0.0) < -1.5:
        hard_fail.append("CPCV_WORST_TOO_LOW")
    if base_run.get("pbo", 0.0) > 0.4:
        hard_fail.append("PBO_TOO_HIGH")
    if summary.get("eval_score", 0.0) < config.EVAL_SCORE_MIN:
        hard_fail.append("EVAL_SCORE_TOO_LOW")

    exp_id = str(uuid4())
    artifact = ArtifactBundle(
        label_config=base_run.get("label_config", {}),
        direction_model=base_run.get("final_guard"),
        risk_model=None,
        calibration_metrics={},
        metadata={
            "genome": policy_spec.feature_genome,
            "risk_budget": policy_spec.risk_budget,
            "eval_summary": summary,
        },
        backtest_results=backtest_results,
    )

    trade_logic = base_run.get("trade_logic") or {}
    verdict_dump = dict(trade_logic)
    verdict_dump["eval"] = summary

    ledger_record = LedgerRecord(
        exp_id=exp_id,
        timestamp=time.time(),
        policy_spec=policy_spec,
        data_hash=_hash_payload(df.values.tobytes()),
        feature_hash=_hash_payload(X_features.columns.tolist()),
        label_hash=base_run.get("label_hash", ""),
        model_artifact_ref="",
        cpcv_metrics=cpcv_metrics,
        pbo=base_run.get("pbo", 0.0),
        risk_report={"ok": True},
        reason_codes=hard_fail,
        verdict_dump=verdict_dump,
    )

    return EvalResult(
        policy_spec=policy_spec,
        score=score,
        summary=summary,
        record=ledger_record,
        artifact=artifact,
    )
