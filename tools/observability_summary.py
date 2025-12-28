import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p10": 0.0, "p90": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _clip_recent(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return rows
    return rows[-limit:] if len(rows) > limit else rows


def _episode_stats(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(episodes)
    cycle_zero = [e for e in episodes if int(e.get("cycle_count", 0)) == 0]
    single_trade = [e for e in episodes if int(e.get("trade_count", 0)) == 1]

    zero_reasons = Counter()
    for e in cycle_zero:
        zero_reasons[e.get("cycle_zero_reason", "UNKNOWN")] += 1

    tag_counts = Counter()
    for e in single_trade:
        for tag in e.get("classification_tags", []):
            tag_counts[tag] += 1

    entry_rates = [float(e.get("entry_signal_rate", 0.0)) for e in episodes]
    gate_distances: Dict[str, List[float]] = {}
    for e in episodes:
        gate = e.get("gate", {}) or {}
        distances = gate.get("distances", {}) or {}
        for code, value in distances.items():
            gate_distances.setdefault(code, []).append(float(value))

    gate_distance_stats = {}
    for code, values in gate_distances.items():
        gate_distance_stats[code] = {
            **_summary_stats(values),
            "fail_rate": float(len([v for v in values if v > 0.0]) / len(values)) if values else 0.0,
        }
    perf_keys = [
        "total_return_pct",
        "excess_return",
        "mdd_pct",
        "sharpe",
        "win_rate",
        "profit_factor",
        "trades_per_year",
    ]
    perf_stats: Dict[str, Dict[str, float]] = {}
    perf_samples = []
    for e in episodes:
        perf = e.get("performance", {})
        if not perf:
            continue
        perf_values = [float(perf.get(k, 0.0)) for k in perf_keys]
        if any(abs(v) > 1e-9 for v in perf_values):
            perf_samples.append(perf)

    for key in perf_keys:
        values = [float(p.get(key, 0.0)) for p in perf_samples]
        perf_stats[key] = _summary_stats(values)

    return {
        "episode_count": total,
        "cycle_zero_count": len(cycle_zero),
        "cycle_zero_rate": float(len(cycle_zero) / total) if total else 0.0,
        "cycle_zero_reasons": dict(zero_reasons),
        "single_trade_count": len(single_trade),
        "single_trade_rate": float(len(single_trade) / total) if total else 0.0,
        "single_trade_tags": dict(tag_counts),
        "entry_signal_rate": _summary_stats(entry_rates),
        "gate_distance_stats": gate_distance_stats,
        "performance_samples": len(perf_samples),
        "performance": perf_stats,
    }


def _batch_stats(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(batches)
    gate_pass = [float(b.get("gate_pass_rate", 0.0)) for b in batches]
    no_cycle = [float(b.get("no_cycle_fail_rate", 0.0)) for b in batches]
    reward_var = [float(b.get("reward_variance", 0.0)) for b in batches]
    agent_exit = [float(b.get("agent_exit_ratio_mean", 0.0)) for b in batches]
    cycle_medians = [float(b.get("cycle_count_hist", {}).get("median", 0.0)) for b in batches]

    fail_counts = Counter()
    nearest_counts = Counter()
    for b in batches:
        fail_counts.update(b.get("gate_fail_reasons", {}))
        nearest_counts.update(b.get("nearest_gate_counts", {}))

    perf_summary = Counter()
    for b in batches:
        perf = b.get("performance_summary", {})
        if perf:
            perf_summary.update({"with_perf": 1})

    return {
        "batch_count": total,
        "gate_pass_rate": _summary_stats(gate_pass),
        "no_cycle_fail_rate": _summary_stats(no_cycle),
        "reward_variance": _summary_stats(reward_var),
        "agent_exit_ratio_mean": _summary_stats(agent_exit),
        "cycle_count_median": _summary_stats(cycle_medians),
        "gate_fail_reasons": dict(fail_counts),
        "nearest_gate_counts": dict(nearest_counts),
        "batches_with_performance_summary": int(perf_summary.get("with_perf", 0)),
    }


def _diagnose(episode_stats: Dict[str, Any], batch_stats: Dict[str, Any]) -> Dict[str, Any]:
    findings: List[str] = []
    actions: List[str] = []

    if int(batch_stats.get("batches_with_performance_summary", 0)) == 0:
        findings.append("Batch performance_summary is missing (new logging not yet populated).")
        actions.append("Run a few new batches to populate performance_summary and topk_performance fields.")

    cycle_zero_rate = float(episode_stats.get("cycle_zero_rate", 0.0))
    zero_reasons = episode_stats.get("cycle_zero_reasons", {})
    single_tags = episode_stats.get("single_trade_tags", {})

    if cycle_zero_rate > 0.2 and zero_reasons.get("NO_ENTRY_ACTION", 0) > 0:
        findings.append("Cycle 0 episodes are dominated by NO_ENTRY_ACTION (entry signal degeneracy).")
        actions.append("Relax entry conditions or reduce logic-tree complexity to raise entry signal rate.")

    if single_tags:
        if single_tags.get("RARE_SIGNAL", 0) >= max(1, int(0.7 * sum(single_tags.values()))):
            findings.append("Single-trade episodes are mostly tagged RARE_SIGNAL.")
            actions.append("Lower entry thresholds or reduce AND terms in early stages.")

    nearest = batch_stats.get("nearest_gate_counts", {})
    if nearest:
        top_nearest = max(nearest.items(), key=lambda kv: kv[1])[0]
        if top_nearest == "FAIL_LOW_RETURN":
            findings.append("Nearest gate is dominated by FAIL_LOW_RETURN (alpha shortfall).")
            actions.append("Check excess_return vs alpha_floor; adjust benchmark/cost model or stage alpha threshold.")
        if top_nearest == "FAIL_MIN_TRADES":
            findings.append("Nearest gate dominated by FAIL_MIN_TRADES (activity shortfall).")
            actions.append("Increase entry signal rate or relax min trades per year in early stage.")

    perf = episode_stats.get("performance", {})
    total_ret = float(perf.get("total_return_pct", {}).get("mean", 0.0))
    excess_ret = float(perf.get("excess_return", {}).get("mean", 0.0))
    if total_ret > 0 and excess_ret < 0:
        findings.append("Total return is positive but excess_return is negative on average.")
        actions.append("Alpha calculation likely penalizes benchmark/cost; verify benchmark ROI and cost config.")

    gate_stats = episode_stats.get("gate_distance_stats", {})
    min_trades_fail = gate_stats.get("FAIL_MIN_TRADES", {}).get("fail_rate", 0.0)
    low_return_fail = gate_stats.get("FAIL_LOW_RETURN", {}).get("fail_rate", 0.0)
    invalid_action_fail = gate_stats.get("FAIL_INVALID_ACTION", {}).get("fail_rate", 0.0)
    if min_trades_fail > 0.15:
        findings.append("FAIL_MIN_TRADES distance is triggered often (entry activity shortfall).")
        actions.append("Reduce min_trades_per_year in early stages or raise entry signal rate.")
    if low_return_fail > 0.3:
        findings.append("FAIL_LOW_RETURN distance is triggered frequently (alpha shortfall).")
        actions.append("Align alpha_floor with benchmark/cost reality or improve excess_return driver.")
    if invalid_action_fail > 0.2:
        findings.append("FAIL_INVALID_ACTION distance is triggered frequently (state-action mismatch).")
        actions.append("Penalize invalid actions or add state-aware action gating in policy outputs.")

    return {
        "findings": findings,
        "actions": actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize observability logs and performance.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs") / "observability",
        help="Directory containing episode_reports.jsonl and batch_reports.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs") / "observability" / "analysis_summary.json",
        help="Path to save the analysis summary JSON.",
    )
    args = parser.parse_args()

    episode_path = args.log_dir / "episode_reports.jsonl"
    batch_path = args.log_dir / "batch_reports.jsonl"

    episodes = _clip_recent(_load_jsonl(episode_path), args.episodes)
    batches = _clip_recent(_load_jsonl(batch_path), args.batches)

    episode_stats = _episode_stats(episodes)
    batch_stats = _batch_stats(batches)
    diagnosis = _diagnose(episode_stats, batch_stats)

    payload = {
        "source": {
            "episodes_path": str(episode_path),
            "batches_path": str(batch_path),
            "episodes_window": args.episodes,
            "batches_window": args.batches,
        },
        "episode_stats": episode_stats,
        "batch_stats": batch_stats,
        "diagnosis": diagnosis,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Saved summary to {args.output}")

    if diagnosis["findings"]:
        print("[Findings]")
        for item in diagnosis["findings"]:
            print(f"- {item}")
    if diagnosis["actions"]:
        print("[Actions]")
        for item in diagnosis["actions"]:
            print(f"- {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
