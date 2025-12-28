from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import config
from src.shared.logger import get_logger

logger = get_logger("orchestration.regression_monitor")


@dataclass
class RegressionSignal:
    code: str
    triggered: bool
    detail: str


@dataclass
class RegressionReport:
    timestamp: float
    batch_id: int
    stage: int
    signals: List[RegressionSignal]
    evidence: Dict[str, Any]


class RegressionMonitor:
    """
    Batch-level regression detector with evidence bundles.
    """

    def __init__(self, incident_dir: Optional[Path] = None):
        self.incident_dir = incident_dir or config.INCIDENT_DIR

    def evaluate(
        self,
        batch_report: Dict[str, Any],
        episode_summaries: List[Dict[str, Any]],
        diversity_info: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[RegressionReport]:
        signals = self._detect_signals(batch_report, diversity_info, history or [])
        triggered = [s for s in signals if s.triggered]
        if not triggered:
            return None

        report = RegressionReport(
            timestamp=time.time(),
            batch_id=int(batch_report.get("batch_id", 0)),
            stage=int(getattr(config, "CURRICULUM_CURRENT_STAGE", 1)),
            signals=signals,
            evidence=self._build_evidence(batch_report, episode_summaries, diversity_info or {}),
        )

        self._log_incident(report)
        return report

    def _detect_signals(
        self,
        batch_report: Dict[str, Any],
        diversity_info: Optional[Dict[str, Any]],
        history: List[Dict[str, Any]],
    ) -> List[RegressionSignal]:
        signals: List[RegressionSignal] = []
        r1 = self._check_deadlock(batch_report)
        signals.append(r1)
        r2 = self._check_one_shot(batch_report)
        signals.append(r2)
        r3 = self._check_invalid_action(batch_report)
        signals.append(r3)
        r4 = self._check_selection_distortion(batch_report, diversity_info or {}, history)
        signals.append(r4)
        return signals

    def _check_deadlock(self, report: Dict[str, Any]) -> RegressionSignal:
        cfg = getattr(config, "REGRESSION_R1_DEADLOCK", {})
        median_cycle = float(report.get("cycle_count_hist", {}).get("median", 0.0))
        reward_var = float(report.get("reward_variance", 0.0))
        gate_pass = float(report.get("gate_pass_rate", 0.0))
        triggered = (
            median_cycle <= float(cfg.get("max_median_cycle", 1.0))
            and reward_var <= float(cfg.get("max_reward_variance", 1e-4))
            and gate_pass <= float(cfg.get("max_gate_pass_rate", 0.01))
        )
        detail = f"median_cycle={median_cycle:.2f}, reward_var={reward_var:.6f}, gate_pass={gate_pass:.2f}"
        return RegressionSignal(code="R1_DEADLOCK", triggered=triggered, detail=detail)

    def _check_one_shot(self, report: Dict[str, Any]) -> RegressionSignal:
        cfg = getattr(config, "REGRESSION_R2_ONE_SHOT", {})
        one_shot_rate = float(report.get("one_shot_hold_rate", 0.0))
        median_hold = float(report.get("avg_hold_bars_hist", {}).get("median", 0.0))
        triggered = (
            one_shot_rate >= float(cfg.get("min_one_shot_hold_rate", 0.6))
            or median_hold >= float(cfg.get("min_avg_hold_bars_median", 150.0))
        )
        detail = f"one_shot_rate={one_shot_rate:.2f}, median_hold={median_hold:.2f}"
        return RegressionSignal(code="R2_ONE_SHOT", triggered=triggered, detail=detail)

    def _check_invalid_action(self, report: Dict[str, Any]) -> RegressionSignal:
        cfg = getattr(config, "REGRESSION_R3_INVALID_ACTION", {})
        invalid_rate = float(report.get("invalid_action_rate_mean", 0.0))
        triggered = invalid_rate >= float(cfg.get("min_invalid_action_rate", 0.1))
        detail = f"invalid_action_rate={invalid_rate:.3f}"
        return RegressionSignal(code="R3_INVALID_ACTION", triggered=triggered, detail=detail)

    def _check_selection_distortion(
        self,
        report: Dict[str, Any],
        diversity_info: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> RegressionSignal:
        cfg = getattr(config, "REGRESSION_R4_SELECTION", {})
        collision_rate = float(diversity_info.get("collision_rate", 0.0))
        avg_jaccard = float(diversity_info.get("avg_jaccard", 0.0))
        diversity_mean = max(0.0, 1.0 - avg_jaccard)

        nearest_gate_counts = report.get("nearest_gate_counts", {}) or {}
        nearest_ratio = self._dominant_ratio(nearest_gate_counts)

        dominated = nearest_ratio >= float(cfg.get("max_nearest_gate_ratio", 0.8))
        diversity_low = diversity_mean <= float(cfg.get("min_diversity_mean", 0.2))
        collision_high = collision_rate >= float(cfg.get("max_collision_rate", 0.6))
        no_improvement = self._distance_not_improving(report, history)

        triggered = (collision_high or diversity_low) or (dominated and no_improvement)
        detail = (
            f"collision_rate={collision_rate:.2f}, diversity_mean={diversity_mean:.2f}, "
            f"nearest_ratio={nearest_ratio:.2f}, no_improvement={no_improvement}"
        )
        return RegressionSignal(code="R4_SELECTION", triggered=triggered, detail=detail)

    def _distance_not_improving(self, report: Dict[str, Any], history: List[Dict[str, Any]]) -> bool:
        if not history:
            return False
        current_dist = float(report.get("distance_to_pass_mean", 0.0))
        recent = [float(r.get("distance_to_pass_mean", 0.0)) for r in history[-int(config.REGRESSION_HISTORY_WINDOW):]]
        if not recent:
            return False
        past_mean = float(np.mean(recent))
        improvement = past_mean - current_dist
        return improvement < float(getattr(config, "REGRESSION_DISTANCE_IMPROVEMENT_MIN", 0.0))

    def _dominant_ratio(self, counts: Dict[str, int]) -> float:
        if not counts:
            return 0.0
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        return max(counts.values()) / total

    def _build_evidence(
        self,
        report: Dict[str, Any],
        summaries: List[Dict[str, Any]],
        diversity_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        top_k = int(getattr(config, "OBS_BATCH_TOP_K", 5))
        sorted_summaries = sorted(
            summaries,
            key=lambda s: float(s.get("reward_total", s.get("eval_score", 0.0))),
            reverse=True,
        )
        topk = []
        for s in sorted_summaries[:top_k]:
            gate = s.get("gate", {})
            topk.append({
                "policy_id": s.get("policy_id"),
                "cycle_count": s.get("cycle_count"),
                "trade_count": s.get("trade_count"),
                "reward_total": s.get("reward_total"),
                "soft_gate_score": gate.get("soft_gate_score"),
                "nearest_gate": gate.get("nearest_gate"),
                "classification_tags": s.get("classification_tags", []),
            })

        distances = [float(s.get("gate", {}).get("distance_to_pass", 0.0)) for s in summaries]
        reward_components = {
            "return_component": [],
            "risk_component": [],
            "cost_component": [],
            "frequency_component": [],
            "gate_component": [],
        }
        for s in summaries:
            comps = s.get("reward_components", {})
            for key in reward_components:
                reward_components[key].append(float(comps.get(key, 0.0)))

        reward_stats = {}
        for key, vals in reward_components.items():
            if not vals:
                reward_stats[key] = {"mean": 0.0, "var": 0.0}
                continue
            reward_stats[key] = {
                "mean": float(np.mean(vals)),
                "var": float(np.var(vals)),
            }

        exit_ratios = [s.get("exit_reason_ratio", {}) for s in summaries]
        exit_ratio_mean = {}
        for ratios in exit_ratios:
            for key, val in ratios.items():
                exit_ratio_mean[key] = exit_ratio_mean.get(key, 0.0) + float(val)
        if summaries:
            for key in exit_ratio_mean:
                exit_ratio_mean[key] /= len(summaries)

        return {
            "batch_metrics": {
                "cycle_median": report.get("cycle_count_hist", {}).get("median", 0.0),
                "reward_variance": report.get("reward_variance", 0.0),
                "gate_pass_rate": report.get("gate_pass_rate", 0.0),
            },
            "topk_snapshot": topk,
            "nearest_gate_counts": report.get("nearest_gate_counts", {}),
            "distance_to_pass_stats": {
                "mean": float(np.mean(distances)) if distances else 0.0,
                "p10": float(np.percentile(distances, 10)) if distances else 0.0,
                "p90": float(np.percentile(distances, 90)) if distances else 0.0,
            },
            "reward_decomposition": reward_stats,
            "exit_reason_ratio_mean": exit_ratio_mean,
            "diversity": {
                "avg_jaccard": float(diversity_info.get("avg_jaccard", 0.0)),
                "collision_rate": float(diversity_info.get("collision_rate", 0.0)),
            },
        }

    def _log_incident(self, report: RegressionReport) -> None:
        payload = {
            "timestamp": report.timestamp,
            "batch_id": report.batch_id,
            "stage": report.stage,
            "signals": [s.__dict__ for s in report.signals],
            "evidence": report.evidence,
        }
        try:
            path = self.incident_dir / "regression_alerts.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as exc:
            self._log_error("log_incident", exc, {"incident_dir": str(self.incident_dir)})

    def _log_error(self, context: str, error: Exception, state: Dict[str, Any]) -> None:
        payload = {
            "error_id": f"regression_monitor_{int(time.time())}",
            "timestamp": time.time(),
            "environment": "local",
            "context": context,
            "stack": str(error),
            "root_cause": "unknown",
            "resolution": "",
            "state": state,
        }
        target = Path(config.LOG_DIR) / "errors" / f"{payload['error_id']}.json"
        try:
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("회귀 모니터 오류 로그 기록 실패")
