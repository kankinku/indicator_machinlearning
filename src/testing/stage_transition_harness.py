from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import config
from src.ledger.repo import LedgerRepo
from src.l3_meta.detectors.regime import RegimeDetector
from src.orchestration.evaluation import evaluate_stage
from src.shared.hashing import hash_dataframe
from src.shared.logger import get_logger

logger = get_logger("testing.stage_transition_harness")


def _load_policy_specs(max_policies: int) -> List[Any]:
    repo = LedgerRepo(config.LEDGER_DIR)
    records = [r for r in repo.load_records() if r.policy_spec]
    if not records:
        return []
    records.sort(key=lambda r: (r.cpcv_metrics or {}).get("eval_score", -9999.0), reverse=True)
    return [r.policy_spec for r in records[:max_policies]]

def _log_error(context: str, error: Exception, state: Dict[str, Any]) -> None:
    payload = {
        "error_id": f"stage_transition_harness_{int(time.time())}",
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
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to write stage transition harness error log")


def run_stage_transition_harness(
    df: pd.DataFrame,
    policies: Optional[List[Any]] = None,
    stage_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    original_stage = int(getattr(config, "CURRICULUM_CURRENT_STAGE", 1))
    try:
        policies = policies or _load_policy_specs(int(getattr(config, "EVAL_HARNESS_MAX_POLICIES", 10)))
        if not policies:
            raise ValueError("No policies available for stage transition harness")

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        stage_ids = stage_ids or sorted(config.CURRICULUM_STAGES.keys())
        window_bars = int(getattr(config, "EVAL_HARNESS_WINDOW_BARS", 400))
        if window_bars <= 0:
            raise ValueError("EVAL_HARNESS_WINDOW_BARS must be > 0")

        if len(df) > window_bars:
            df = df.iloc[-window_bars:]

        detector = RegimeDetector()
        regime = detector.detect(df)
        dataset_sig = hash_dataframe(df)

        results_payload: Dict[str, Any] = {
            "timestamp": time.time(),
            "policy_count": len(policies),
            "stage_ids": stage_ids,
            "dataset_sig": dataset_sig,
            "results": [],
        }

        for stage_id in stage_ids:
            stage_id = int(stage_id)
            config.CURRICULUM_CURRENT_STAGE = stage_id
            results, diag = evaluate_stage(
                policies,
                df,
                "full",
                regime.label,
                n_jobs=1,
                stage_id=stage_id,
            )
            eval_usage = {}
            if isinstance(diag, dict):
                eval_usage = diag.get("eval_usage", {})
            results_payload["results"].append({
                "stage_id": stage_id,
                "eval_usage": eval_usage,
                "result_count": len(results),
            })

        out_dir = Path(config.HARNESS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"stage_transition_harness_{int(time.time())}.json"
        out_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
        logger.info(f"[StageHarness] Wrote report to {out_path}")
        return results_payload
    except Exception as exc:
        _log_error("run_stage_transition_harness", exc, {"stage_ids": stage_ids or []})
        raise
    finally:
        config.CURRICULUM_CURRENT_STAGE = original_stage
