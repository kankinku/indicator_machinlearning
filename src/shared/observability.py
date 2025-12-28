"""Cycle = Enter -> Exit -> re-entry possible state (SSOT)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import config
from src.shared.event_bus import record_event
from src.shared.logger import get_logger

logger = get_logger("shared.observability")


@dataclass
class CycleMeta:
    cycle_id: int
    entry_bar_index: int
    exit_bar_index: int
    hold_bars: int
    exit_reason: str
    reentry_gap: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "entry_bar_index": self.entry_bar_index,
            "exit_bar_index": self.exit_bar_index,
            "hold_bars": self.hold_bars,
            "exit_reason": self.exit_reason,
            "reentry_gap": self.reentry_gap,
        }


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_exit_reason(reason: Optional[str]) -> str:
    if not reason:
        return "Other"
    norm = str(reason).strip().lower()
    if norm in {"sl", "stop", "stop_loss"}:
        return "SL"
    if norm in {"tp", "target", "take_profit"}:
        return "TP"
    if norm in {"horizon", "time", "eod", "expiry"}:
        return "HORIZON"
    if norm in {"signal", "agentexit", "agent_exit", "agent exit", "rule"}:
        return "AGENT_EXIT"
    return "Other"


def _percentile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=float), q))


def compute_cycle_stats(
    trades: List[Dict[str, Any]],
    total_bars: int,
    policy_id: Optional[str] = None,
    stage: Optional[str] = None,
) -> Tuple[List[CycleMeta], Dict[str, Any]]:
    # Cycle = Enter -> Exit -> return to re-entry possible state.
    total_bars = int(total_bars or 0)
    last_bar = total_bars - 1
    trades_sorted = sorted(trades, key=lambda t: _safe_int(t.get("entry_idx"), 0) or 0)

    exit_reason_counts = {"SL": 0, "TP": 0, "HORIZON": 0, "AGENT_EXIT": 0, "Other": 0}
    hold_bars_list: List[int] = []
    reentry_gaps: List[int] = []
    flat_gaps: List[int] = []
    cycles: List[CycleMeta] = []
    cycle_id = 0

    for idx, trade in enumerate(trades_sorted):
        entry_idx = _safe_int(trade.get("entry_idx"))
        exit_idx = _safe_int(trade.get("exit_idx"))
        if entry_idx is None or exit_idx is None:
            continue

        hold_bars = _safe_int(trade.get("bars"))
        if hold_bars is None:
            hold_bars = max(0, exit_idx - entry_idx)
        hold_bars_list.append(hold_bars)

        exit_reason = _normalize_exit_reason(trade.get("exit_reason") or trade.get("reason"))
        exit_reason_counts[exit_reason] += 1

        next_entry_idx = None
        if idx + 1 < len(trades_sorted):
            next_entry_idx = _safe_int(trades_sorted[idx + 1].get("entry_idx"))
        reentry_gap = None
        if next_entry_idx is not None:
            reentry_gap = max(0, next_entry_idx - exit_idx)
            reentry_gaps.append(reentry_gap)
            flat_gaps.append(max(0, reentry_gap - 1))

        cycle_complete = (last_bar >= 0) and (exit_idx < last_bar)
        if cycle_complete:
            cycle_id += 1
            cycle = CycleMeta(
                cycle_id=cycle_id,
                entry_bar_index=entry_idx,
                exit_bar_index=exit_idx,
                hold_bars=hold_bars,
                exit_reason=exit_reason,
                reentry_gap=reentry_gap,
            )
            cycles.append(cycle)
            record_event(
                "CYCLE_COMPLETE",
                policy_id=policy_id,
                stage=stage,
                payload=cycle.to_dict(),
            )

    entry_count = len(trades_sorted)
    exit_count = len(trades_sorted)

    flat_bars = total_bars
    if total_bars > 0 and trades_sorted:
        first_entry = _safe_int(trades_sorted[0].get("entry_idx"), 0) or 0
        last_exit = _safe_int(trades_sorted[-1].get("exit_idx"), last_bar) or last_bar
        flat_bars = 0
        flat_bars += max(0, first_entry)
        flat_bars += max(0, (total_bars - 1) - last_exit)
        for gap in flat_gaps:
            flat_bars += gap

    avg_hold_bars = float(np.mean(hold_bars_list)) if hold_bars_list else 0.0
    p95_hold_bars = _percentile(hold_bars_list, 95.0)
    avg_flat_gap = float(np.mean(flat_gaps)) if flat_gaps else 0.0

    exit_reason_ratio = {
        key: (count / exit_count if exit_count else 0.0)
        for key, count in exit_reason_counts.items()
    }

    return cycles, {
        "trade_count": entry_count,
        "cycle_count": len(cycles),
        "entry_count": entry_count,
        "exit_count": exit_count,
        "avg_hold_bars": avg_hold_bars,
        "p95_hold_bars": p95_hold_bars,
        "flat_bars": int(flat_bars),
        "avg_flat_gap": avg_flat_gap,
        "reentry_gaps": reentry_gaps,
        "exit_reason_counts": exit_reason_counts,
        "exit_reason_ratio": exit_reason_ratio,
    }


def classify_episode(
    trade_count: int,
    avg_hold_bars: float,
    total_bars: int,
    entry_rate: float,
    complexity_score: float,
    exit_reason_ratio: Dict[str, float],
) -> Tuple[bool, List[str]]:
    tags: List[str] = []
    one_shot_hold = False
    total_bars = max(1, int(total_bars or 0))
    hold_ratio = float(avg_hold_bars) / float(total_bars)
    horizon_ratio = float(exit_reason_ratio.get("HORIZON", 0.0))

    if trade_count == 1 and (
        hold_ratio >= config.OBS_ONE_SHOT_HOLD_RATIO
        or horizon_ratio >= config.OBS_ONE_SHOT_HOLD_HORIZON_RATIO
    ):
        tags.append("ONE_SHOT_HOLD")
        one_shot_hold = True

    if trade_count <= config.OBS_RARE_SIGNAL_ENTRY_COUNT_MAX or entry_rate <= config.OBS_RARE_SIGNAL_ENTRY_RATE_MAX:
        tags.append("RARE_SIGNAL")

    if entry_rate <= config.OBS_OVERFILTER_ENTRY_RATE_MAX and complexity_score >= config.OBS_OVERFILTER_COMPLEXITY_MIN:
        tags.append("OVER_FILTER")

    return one_shot_hold, sorted(set(tags))


def compute_gate_status(metrics: Any) -> Dict[str, Any]:
    from src.l1_judge.evaluator import collect_validation_failures, compute_gate_diagnostics

    failures = collect_validation_failures(metrics)
    diag = compute_gate_diagnostics(metrics)

    fail_reason = diag.hard_fail_reasons[0] if diag.hard_fail_reasons else "PASS"
    distance_to_pass = 0.0
    if diag.nearest_gate != "PASS":
        distance_to_pass = float(diag.distances.get(diag.nearest_gate, 0.0))

    return {
        "gate_pass": diag.approval_pass,
        "approval_pass": diag.approval_pass,
        "hard_fail_reasons": diag.hard_fail_reasons,
        "fail_reason": fail_reason,
        "distances": diag.distances,
        "distance_to_pass": distance_to_pass,
        "nearest_gate": diag.nearest_gate,
        "soft_gate_score": float(diag.soft_gate_score),
        "failure_codes": [f.code for f in failures],
    }


def build_episode_summary(
    policy_id: str,
    stage: str,
    metrics: Any,
    bt_result: Any,
    reward_breakdown: Optional[Dict[str, Any]],
    eval_score: float,
    module_key: str,
) -> Dict[str, Any]:
    total_bars = int(getattr(metrics, "bars_total", 0) or 0)
    trades = []
    if bt_result is not None:
        trades = getattr(bt_result, "trades", []) or []

    cycles, stats = compute_cycle_stats(trades, total_bars, policy_id=policy_id, stage=stage)

    entry_rate = float(getattr(metrics.trades, "entry_signal_rate", 0.0))
    complexity_score = float(getattr(metrics, "complexity_score", 0.0))
    one_shot_hold, tags = classify_episode(
        trade_count=stats["trade_count"],
        avg_hold_bars=stats["avg_hold_bars"],
        total_bars=total_bars,
        entry_rate=entry_rate,
        complexity_score=complexity_score,
        exit_reason_ratio=stats["exit_reason_ratio"],
    )

    gate = compute_gate_status(metrics)

    reward_total = eval_score
    reward_components = {
        "return_component": 0.0,
        "risk_component": 0.0,
        "cost_component": 0.0,
        "frequency_component": 0.0,
        "gate_component": 0.0,
    }
    if reward_breakdown:
        reward_total = float(reward_breakdown.get("total", eval_score))
        reward_components["return_component"] = float(reward_breakdown.get("return_component", 0.0))
        reward_components["risk_component"] = float(reward_breakdown.get("mdd_penalty", 0.0)) + float(
            reward_breakdown.get("winrate_penalty", 0.0)
        )
        reward_components["cost_component"] = float(reward_breakdown.get("cost_survival_component", 0.0))
        reward_components["frequency_component"] = float(reward_breakdown.get("trades_component", 0.0))
        if reward_breakdown.get("is_rejected"):
            reward_components["gate_component"] = reward_total

    equity = getattr(metrics, "equity", None)
    trades_stats = getattr(metrics, "trades", None)
    performance = {
        "total_return_pct": _safe_float(getattr(equity, "total_return_pct", None)),
        "cagr": _safe_float(getattr(equity, "cagr", None)),
        "sharpe": _safe_float(getattr(equity, "sharpe", None)),
        "mdd_pct": _safe_float(getattr(equity, "max_drawdown_pct", None)),
        "vol_pct": _safe_float(getattr(equity, "vol_pct", None)),
        "exposure_ratio": _safe_float(getattr(equity, "exposure_ratio", None)),
        "percent_in_market": _safe_float(getattr(equity, "percent_in_market", None)),
        "excess_return": _safe_float(getattr(equity, "excess_return", None)),
        "win_rate": _safe_float(getattr(trades_stats, "win_rate", None)),
        "profit_factor": _safe_float(getattr(trades_stats, "profit_factor", None)),
        "expectancy": _safe_float(getattr(trades_stats, "expectancy", None)),
        "reward_risk": _safe_float(getattr(trades_stats, "reward_risk", None)),
        "trades_per_year": _safe_float(getattr(trades_stats, "trades_per_year", None)),
        "avg_holding_bars": _safe_float(getattr(trades_stats, "avg_holding_bars", None)),
    }

    cycle_zero_reason = None
    if stats["cycle_count"] == 0:
        if stats["trade_count"] == 0:
            cycle_zero_reason = "NO_ENTRY_ACTION"
        elif stats["exit_reason_ratio"].get("AGENT_EXIT", 0.0) == 0.0:
            cycle_zero_reason = "NO_EXIT_ACTION"
        else:
            cycle_zero_reason = "NO_REENTRY_WINDOW"

    return {
        "timestamp": time.time(),
        "policy_id": policy_id,
        "stage": stage,
        "module_key": module_key,
        "trade_count": stats["trade_count"],
        "cycle_count": stats["cycle_count"],
        "entry_count": stats["entry_count"],
        "exit_count": stats["exit_count"],
        "avg_hold_bars": stats["avg_hold_bars"],
        "p95_hold_bars": stats["p95_hold_bars"],
        "flat_bars": stats["flat_bars"],
        "avg_flat_gap": stats["avg_flat_gap"],
        "reentry_gaps": stats["reentry_gaps"],
        "entry_signal_rate": entry_rate,
        "invalid_action_count": int(getattr(metrics.trades, "invalid_action_count", 0)),
        "invalid_action_rate": float(getattr(metrics.trades, "invalid_action_rate", 0.0)),
        "exit_reason_ratio": stats["exit_reason_ratio"],
        "cycle_zero_reason": cycle_zero_reason,
        "one_shot_hold": one_shot_hold,
        "classification_tags": tags,
        "reward_total": reward_total,
        "reward_components": reward_components,
        "reward_breakdown": reward_breakdown or {},
        "performance": performance,
        "gate": gate,
        "cycle_metadata": [c.to_dict() for c in cycles],
        "eval_score": float(eval_score),
    }


def _histogram(values: List[int], bins: List[int]) -> Dict[str, Any]:
    if not values:
        return {
            "bins": bins,
            "counts": [0] * (len(bins) - 1),
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
        }
    arr = np.array(values, dtype=float)
    counts, _ = np.histogram(arr, bins=bins)
    return {
        "bins": bins,
        "counts": counts.tolist(),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p10": 0.0, "p90": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def build_batch_report(
    batch_id: int,
    summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cycle_counts = [int(s.get("cycle_count", 0)) for s in summaries]
    trade_counts = [int(s.get("trade_count", 0)) for s in summaries]
    avg_hold_bars = [float(s.get("avg_hold_bars", 0.0)) for s in summaries]
    invalid_action_rates = [float(s.get("invalid_action_rate", 0.0)) for s in summaries]
    gate_pass_rate = float(np.mean([1.0 if s.get("gate", {}).get("gate_pass") else 0.0 for s in summaries])) if summaries else 0.0
    reward_scores = [float(s.get("reward_total", s.get("eval_score", 0.0))) for s in summaries]
    reward_variance = float(np.var(reward_scores)) if reward_scores else 0.0
    soft_gate_scores = [float(s.get("gate", {}).get("soft_gate_score", 0.0)) for s in summaries]
    one_shot_hold_rate = float(np.mean([1.0 if s.get("one_shot_hold") else 0.0 for s in summaries])) if summaries else 0.0
    agent_exit_ratios = [float(s.get("exit_reason_ratio", {}).get("AGENT_EXIT", 0.0)) for s in summaries]
    cost_components = [float(s.get("reward_components", {}).get("cost_component", 0.0)) for s in summaries]
    single_trade = [s for s in summaries if int(s.get("trade_count", 0)) == 1]
    single_trade_rate = float(len(single_trade) / len(summaries)) if summaries else 0.0
    single_trade_entry_rates = [float(s.get("entry_signal_rate", 0.0)) for s in single_trade]
    single_trade_entry_rate_mean = float(np.mean(single_trade_entry_rates)) if single_trade_entry_rates else 0.0
    single_trade_tags: Dict[str, int] = {}
    for s in single_trade:
        for tag in s.get("classification_tags", []):
            single_trade_tags[tag] = single_trade_tags.get(tag, 0) + 1
    single_trade_exit_sum = {"SL": 0.0, "TP": 0.0, "HORIZON": 0.0, "AGENT_EXIT": 0.0, "Other": 0.0}
    for s in single_trade:
        ratios = s.get("exit_reason_ratio", {})
        for key in single_trade_exit_sum:
            single_trade_exit_sum[key] += float(ratios.get(key, 0.0))
    single_trade_exit_ratio = {k: (v / len(single_trade) if single_trade else 0.0) for k, v in single_trade_exit_sum.items()}

    fail_reason_counts: Dict[str, int] = {}
    nearest_gate_counts: Dict[str, int] = {}
    no_cycle_fails = 0
    distance_values: List[float] = []
    for s in summaries:
        gate = s.get("gate", {})
        fail_reason = gate.get("fail_reason", "PASS")
        if fail_reason != "PASS":
            fail_reason_counts[fail_reason] = fail_reason_counts.get(fail_reason, 0) + 1
        hard_reasons = gate.get("hard_fail_reasons", [])
        if "FAIL_NO_CYCLE" in hard_reasons or fail_reason == "FAIL_NO_CYCLE":
            no_cycle_fails += 1
        distance_values.append(float(gate.get("distance_to_pass", 0.0)))
        nearest_gate = gate.get("nearest_gate", "PASS")
        if nearest_gate != "PASS":
            nearest_gate_counts[nearest_gate] = nearest_gate_counts.get(nearest_gate, 0) + 1

    top_k = int(getattr(config, "OBS_BATCH_TOP_K", 5))
    eligible = [s for s in summaries if int(s.get("cycle_count", 0)) > 0]
    eligible.sort(key=lambda s: float(s.get("reward_total", s.get("eval_score", 0.0))), reverse=True)
    topk = eligible[:top_k]

    topk_unique = []
    seen_ids = set()
    for s in topk:
        policy_id = s.get("policy_id")
        if policy_id in seen_ids:
            continue
        seen_ids.add(policy_id)
        topk_unique.append(s)

    top_tags: Dict[str, int] = {}
    for s in topk_unique:
        for tag in s.get("classification_tags", []):
            top_tags[tag] = top_tags.get(tag, 0) + 1

    exit_ratio_sum = {"SL": 0.0, "TP": 0.0, "HORIZON": 0.0, "AGENT_EXIT": 0.0, "Other": 0.0}
    for s in topk_unique:
        ratios = s.get("exit_reason_ratio", {})
        for key in exit_ratio_sum:
            exit_ratio_sum[key] += float(ratios.get(key, 0.0))
    topk_exit_ratio = {k: (v / len(topk_unique) if topk_unique else 0.0) for k, v in exit_ratio_sum.items()}

    perf_keys = [
        "total_return_pct",
        "mdd_pct",
        "sharpe",
        "win_rate",
        "profit_factor",
        "trades_per_year",
        "expectancy",
        "reward_risk",
        "excess_return",
    ]
    performance_summary: Dict[str, Dict[str, float]] = {}
    topk_performance_summary: Dict[str, Dict[str, float]] = {}
    for key in perf_keys:
        values = [float(s.get("performance", {}).get(key, 0.0)) for s in summaries]
        performance_summary[key] = _summary_stats(values)
        topk_values = [float(s.get("performance", {}).get(key, 0.0)) for s in topk_unique]
        topk_performance_summary[key] = _summary_stats(topk_values)

    topk_performance = []
    for s in topk_unique:
        perf = s.get("performance", {})
        gate = s.get("gate", {})
        topk_performance.append({
            "policy_id": s.get("policy_id"),
            "module_key": s.get("module_key"),
            "reward_total": float(s.get("reward_total", s.get("eval_score", 0.0))),
            "eval_score": float(s.get("eval_score", 0.0)),
            "total_return_pct": float(perf.get("total_return_pct", 0.0)),
            "excess_return": float(perf.get("excess_return", 0.0)),
            "mdd_pct": float(perf.get("mdd_pct", 0.0)),
            "sharpe": float(perf.get("sharpe", 0.0)),
            "profit_factor": float(perf.get("profit_factor", 0.0)),
            "trades_per_year": float(perf.get("trades_per_year", 0.0)),
            "trade_count": int(s.get("trade_count", 0)),
            "cycle_count": int(s.get("cycle_count", 0)),
            "gate_pass": bool(gate.get("gate_pass")),
            "nearest_gate": gate.get("nearest_gate", "PASS"),
        })

    return {
        "timestamp": time.time(),
        "batch_id": batch_id,
        "episode_count": len(summaries),
        "gate_pass_rate": gate_pass_rate,
        "reward_variance": reward_variance,
        "cycle_count_hist": _histogram(cycle_counts, config.OBS_CYCLE_COUNT_BINS),
        "trade_count_hist": _histogram(trade_counts, config.OBS_TRADE_COUNT_BINS),
        "avg_hold_bars_hist": _histogram([int(v) for v in avg_hold_bars], config.OBS_HOLD_BARS_BINS),
        "no_cycle_fail_rate": float(no_cycle_fails / len(summaries)) if summaries else 0.0,
        "gate_fail_reasons": fail_reason_counts,
        "distance_to_pass_hist": _histogram([int(v) for v in distance_values], config.OBS_TRADE_COUNT_BINS),
        "distance_to_pass_mean": float(np.mean(distance_values)) if distance_values else 0.0,
        "invalid_action_rate_mean": float(np.mean(invalid_action_rates)) if invalid_action_rates else 0.0,
        "invalid_action_rate_p90": float(np.percentile(invalid_action_rates, 90)) if invalid_action_rates else 0.0,
        "soft_gate_score_mean": float(np.mean(soft_gate_scores)) if soft_gate_scores else 0.0,
        "soft_gate_score_p10": float(np.percentile(soft_gate_scores, 10)) if soft_gate_scores else 0.0,
        "soft_gate_score_p90": float(np.percentile(soft_gate_scores, 90)) if soft_gate_scores else 0.0,
        "one_shot_hold_rate": one_shot_hold_rate,
        "single_trade_rate": single_trade_rate,
        "single_trade_entry_rate_mean": single_trade_entry_rate_mean,
        "single_trade_tags": single_trade_tags,
        "single_trade_exit_reason_ratio": single_trade_exit_ratio,
        "agent_exit_ratio_mean": float(np.mean(agent_exit_ratios)) if agent_exit_ratios else 0.0,
        "reward_component_means": {
            "cost_component": float(np.mean(cost_components)) if cost_components else 0.0,
        },
        "topk_tags": top_tags,
        "topk_exit_reason_ratio": topk_exit_ratio,
        "performance_summary": performance_summary,
        "topk_performance_summary": topk_performance_summary,
        "topk_performance": topk_performance,
        "nearest_gate_counts": nearest_gate_counts,
    }


def detect_deadlock(batch_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    window = int(getattr(config, "OBS_DEADLOCK_WINDOW", 20))
    recent = batch_reports[-window:] if batch_reports else []
    if not recent:
        return {"deadlock": False, "reason": "NO_HISTORY"}

    medians = [float(r.get("cycle_count_hist", {}).get("median", 0.0)) for r in recent]
    gate_pass = [float(r.get("gate_pass_rate", 0.0)) for r in recent]
    reward_var = [float(r.get("reward_variance", 0.0)) for r in recent]

    median_cycle = float(np.median(medians)) if medians else 0.0
    mean_gate_pass = float(np.mean(gate_pass)) if gate_pass else 0.0
    mean_reward_var = float(np.mean(reward_var)) if reward_var else 0.0

    deadlock = (
        median_cycle <= float(getattr(config, "OBS_DEADLOCK_MEDIAN_CYCLES_MAX", 1.0))
        and mean_reward_var <= float(getattr(config, "OBS_DEADLOCK_REWARD_VARIANCE_MAX", 1e-4))
        and mean_gate_pass <= float(getattr(config, "OBS_DEADLOCK_GATE_PASS_MAX", 0.01))
    )

    return {
        "deadlock": deadlock,
        "median_cycle": median_cycle,
        "mean_reward_variance": mean_reward_var,
        "mean_gate_pass_rate": mean_gate_pass,
    }


def _write_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def log_episode(summary: Dict[str, Any]) -> None:
    try:
        target = Path(config.OBSERVABILITY_DIR) / "episode_reports.jsonl"
        _write_jsonl(target, summary)
    except Exception as exc:
        _log_observability_error("episode_log", exc, {"policy_id": summary.get("policy_id")})


def log_batch(report: Dict[str, Any]) -> None:
    try:
        target = Path(config.OBSERVABILITY_DIR) / "batch_reports.jsonl"
        _write_jsonl(target, report)
    except Exception as exc:
        _log_observability_error("batch_log", exc, {"batch_id": report.get("batch_id")})


def load_recent_batch_reports(limit: int) -> List[Dict[str, Any]]:
    path = Path(config.OBSERVABILITY_DIR) / "batch_reports.jsonl"
    if not path.exists():
        return []
    reports: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                reports.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return reports[-limit:] if limit > 0 else reports


def _log_observability_error(context: str, error: Exception, state: Dict[str, Any]) -> None:
    log_dir = Path(config.LOG_DIR) / "errors"
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "error_id": f"observability_{int(time.time())}",
        "timestamp": time.time(),
        "environment": "local",
        "context": context,
        "stack": str(error),
        "root_cause": "unknown",
        "resolution": "",
        "state": state,
    }
    target = log_dir / f"{payload['error_id']}.json"
    try:
        target.write_text(json.dumps(payload, indent=2))
    except Exception:
        logger.exception("Failed to write observability error log")
