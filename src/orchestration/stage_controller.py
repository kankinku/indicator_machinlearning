from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import config
from src.l3_meta.curriculum_controller import get_curriculum_controller
from src.shared.logger import get_logger

logger = get_logger("orchestration.stage_controller")


@dataclass
class StageAutoState:
    current_stage: int = 1
    last_change_batch: int = 0
    promote_streak: int = 0
    demote_streak: int = 0


@dataclass
class StageDecision:
    stage_before: int
    stage_after: int
    action: str
    reasons: List[str]
    metrics: Dict[str, Any]


class StageController:
    """
    Batch-driven Stage Controller.
    Uses batch_reports metrics as SSOT for promotion/demotion decisions.
    """

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or (config.LEDGER_DIR / "stage_auto_state.json")
        self.state = self._load_state()
        curriculum = get_curriculum_controller()
        if self.state.current_stage != curriculum.current_stage:
            self.state.current_stage = curriculum.current_stage
            self._save_state()

    def update(self, batch_report: Dict[str, Any], batch_id: int) -> StageDecision:
        if not getattr(config, "STAGE_AUTO_ENABLED", True):
            return StageDecision(
                stage_before=self.state.current_stage,
                stage_after=self.state.current_stage,
                action="disabled",
                reasons=[],
                metrics=self._extract_metrics(batch_report),
            )

        if batch_id - self.state.last_change_batch < int(getattr(config, "STAGE_AUTO_MIN_BATCH_INTERVAL", 0)):
            return StageDecision(
                stage_before=self.state.current_stage,
                stage_after=self.state.current_stage,
                action="cooldown",
                reasons=[],
                metrics=self._extract_metrics(batch_report),
            )

        stage_before = self.state.current_stage
        max_stage = max(config.CURRICULUM_STAGES.keys())
        min_stage = min(config.CURRICULUM_STAGES.keys())

        demote_rules = getattr(config, "STAGE_AUTO_DEMOTION_RULES", {}).get(stage_before, {})
        promote_rules = getattr(config, "STAGE_AUTO_PROMOTION_RULES", {}).get(stage_before, {})

        demote_triggered, demote_reasons = self._evaluate_rules(batch_report, demote_rules)
        promote_ready, promote_reasons = self._evaluate_rules(batch_report, promote_rules)

        if demote_triggered:
            self.state.demote_streak += 1
        else:
            self.state.demote_streak = 0

        if promote_ready:
            self.state.promote_streak += 1
        else:
            self.state.promote_streak = 0

        action = "hold"
        reasons: List[str] = []
        stage_after = stage_before

        demote_streak = int(getattr(config, "STAGE_AUTO_DEMOTE_STREAK", 1))
        promote_streak = int(getattr(config, "STAGE_AUTO_PROMOTE_STREAK", 1))

        if demote_triggered and stage_before > min_stage and self.state.demote_streak >= demote_streak:
            stage_after = stage_before - 1
            action = "demote"
            reasons = demote_reasons
            self._apply_stage_change(stage_after, action, reasons, batch_id)
        elif promote_ready and stage_before < max_stage and self.state.promote_streak >= promote_streak:
            stage_after = stage_before + 1
            action = "promote"
            reasons = promote_reasons
            self._apply_stage_change(stage_after, action, reasons, batch_id)

        self._save_state()

        return StageDecision(
            stage_before=stage_before,
            stage_after=stage_after,
            action=action,
            reasons=reasons,
            metrics=self._extract_metrics(batch_report),
        )

    def _apply_stage_change(self, stage_after: int, action: str, reasons: List[str], batch_id: int) -> None:
        self.state.current_stage = stage_after
        self.state.last_change_batch = batch_id
        self.state.promote_streak = 0
        self.state.demote_streak = 0

        config.CURRICULUM_CURRENT_STAGE = stage_after
        curriculum = get_curriculum_controller()
        curriculum.set_stage(
            stage_after,
            reason=f"auto_{action}",
            batch_id=batch_id,
            metadata={"reasons": reasons},
        )

        action_label = "승격" if action == "promote" else "강등" if action == "demote" else action.upper()
        logger.info(f"[Stage] {action_label} -> Stage {stage_after} (batch={batch_id})")

    def _extract_metrics(self, report: Dict[str, Any]) -> Dict[str, Any]:
        cycle_hist = report.get("cycle_count_hist", {})
        hold_hist = report.get("avg_hold_bars_hist", {})
        nearest_gate_counts = report.get("nearest_gate_counts", {})
        nearest_gate = self._top_key(nearest_gate_counts)

        return {
            "median_cycle": float(cycle_hist.get("median", 0.0)),
            "median_hold_bars": float(hold_hist.get("median", 0.0)),
            "no_cycle_fail_rate": float(report.get("no_cycle_fail_rate", 0.0)),
            "reward_variance": float(report.get("reward_variance", 0.0)),
            "agent_exit_ratio": float(report.get("agent_exit_ratio_mean", 0.0)),
            "invalid_action_rate": float(report.get("invalid_action_rate_mean", 0.0)),
            "gate_pass_rate": float(report.get("gate_pass_rate", 0.0)),
            "cost_component_mean": float(report.get("reward_component_means", {}).get("cost_component", 0.0)),
            "distance_to_pass_mean": float(report.get("distance_to_pass_mean", 0.0)),
            "wf_pass_rate": float(report.get("wf_pass_rate", 0.0)),
            "nearest_gate": nearest_gate,
        }

    def _evaluate_rules(self, report: Dict[str, Any], rules: Dict[str, Any]) -> Tuple[bool, List[str]]:
        if not rules:
            return True, []

        metrics = self._extract_metrics(report)
        failures = []

        def fail(msg: str) -> None:
            failures.append(msg)

        for key, threshold in rules.items():
            if key == "min_median_cycle":
                if metrics["median_cycle"] < float(threshold):
                    fail(f"median_cycle {metrics['median_cycle']:.2f} < {threshold}")
            elif key == "max_no_cycle_fail_rate":
                if metrics["no_cycle_fail_rate"] > float(threshold):
                    fail(f"no_cycle_fail_rate {metrics['no_cycle_fail_rate']:.2f} > {threshold}")
            elif key == "min_reward_variance":
                if metrics["reward_variance"] < float(threshold):
                    fail(f"reward_variance {metrics['reward_variance']:.6f} < {threshold}")
            elif key == "min_agent_exit_ratio":
                if metrics["agent_exit_ratio"] < float(threshold):
                    fail(f"agent_exit_ratio {metrics['agent_exit_ratio']:.2f} < {threshold}")
            elif key == "max_invalid_action_rate":
                if metrics["invalid_action_rate"] > float(threshold):
                    fail(f"invalid_action_rate {metrics['invalid_action_rate']:.2f} > {threshold}")
            elif key == "max_median_hold_bars":
                if metrics["median_hold_bars"] > float(threshold):
                    fail(f"median_hold_bars {metrics['median_hold_bars']:.2f} > {threshold}")
            elif key == "min_cost_component_mean":
                if metrics["cost_component_mean"] < float(threshold):
                    fail(f"cost_component_mean {metrics['cost_component_mean']:.4f} < {threshold}")
            elif key == "min_gate_pass_rate":
                if metrics["gate_pass_rate"] < float(threshold):
                    fail(f"gate_pass_rate {metrics['gate_pass_rate']:.2f} < {threshold}")
            elif key == "max_distance_to_pass_mean":
                if metrics["distance_to_pass_mean"] > float(threshold):
                    fail(f"distance_to_pass_mean {metrics['distance_to_pass_mean']:.2f} > {threshold}")
            elif key == "min_wf_pass_rate":
                if metrics["wf_pass_rate"] < float(threshold):
                    fail(f"wf_pass_rate {metrics['wf_pass_rate']:.2f} < {threshold}")
            elif key == "nearest_gate_not_in":
                banned = set(threshold or [])
                if metrics["nearest_gate"] in banned:
                    fail(f"nearest_gate {metrics['nearest_gate']} in {sorted(banned)}")
            else:
                fail(f"unknown_rule {key}")

        return len(failures) == 0, failures

    def _top_key(self, counts: Dict[str, int]) -> str:
        if not counts:
            return "PASS"
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _load_state(self) -> StageAutoState:
        if not self.state_file.exists():
            return StageAutoState(current_stage=get_curriculum_controller().current_stage)
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            return StageAutoState(
                current_stage=int(payload.get("current_stage", 1)),
                last_change_batch=int(payload.get("last_change_batch", 0)),
                promote_streak=int(payload.get("promote_streak", 0)),
                demote_streak=int(payload.get("demote_streak", 0)),
            )
        except Exception as exc:
            self._log_error("load_state", exc, {"state_file": str(self.state_file)})
            return StageAutoState(current_stage=get_curriculum_controller().current_stage)

    def _save_state(self) -> None:
        payload = asdict(self.state)
        try:
            self.state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            self._log_error("save_state", exc, {"state_file": str(self.state_file)})

    def _log_error(self, context: str, error: Exception, state: Dict[str, Any]) -> None:
        payload = {
            "error_id": f"stage_controller_{int(time.time())}",
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
            logger.exception("스테이지 컨트롤러 오류 로그 기록 실패")
