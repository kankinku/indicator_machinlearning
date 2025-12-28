from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import config
from src.ledger.repo import LedgerRepo
from src.orchestration.evaluation import evaluate_stage
from src.l3_meta.detectors.regime import RegimeDetector
from src.shared.hashing import hash_dataframe
from src.shared.logger import get_logger

logger = get_logger("testing.eval_harness")


@dataclass
class ScenarioWindow:
    name: str
    start: int
    end: int
    start_ts: str
    end_ts: str
    score: float


def _window_metrics(close: pd.Series, window_bars: int) -> pd.DataFrame:
    returns = close.pct_change().fillna(0.0)
    window_return = (close / close.shift(window_bars) - 1.0).abs()
    window_vol = returns.rolling(window_bars).std()
    return pd.DataFrame({
        "abs_return": window_return,
        "vol": window_vol,
    }).dropna()


def _select_index(metrics: pd.DataFrame, mode: str) -> Optional[int]:
    if metrics.empty:
        return None
    if mode == "trend":
        return int(metrics["abs_return"].idxmax())
    if mode == "range":
        # favor low return and low vol
        score = -(metrics["abs_return"] + metrics["vol"])
        return int(score.idxmax())
    if mode == "volatile":
        return int(metrics["vol"].idxmax())
    return None


def build_scenario_windows(
    df: pd.DataFrame,
    window_bars: int,
    scenarios: Dict[str, Dict[str, Any]],
) -> Dict[str, ScenarioWindow]:
    if df.empty or window_bars <= 0 or len(df) < window_bars:
        return {}
    if "close" not in df.columns:
        raise ValueError("Missing 'close' column for scenario selection")

    metrics = _window_metrics(df["close"], window_bars)
    windows: Dict[str, ScenarioWindow] = {}
    for name, cfg in scenarios.items():
        mode = cfg.get("score", name)
        idx = _select_index(metrics, mode)
        if idx is None:
            continue
        end = idx
        start = end - window_bars + 1
        start = max(0, start)
        end = min(len(df) - 1, end)
        start_ts = str(df.index[start])
        end_ts = str(df.index[end])
        score = float(metrics.loc[end, "abs_return"]) if end in metrics.index else 0.0
        windows[name] = ScenarioWindow(
            name=name,
            start=start,
            end=end,
            start_ts=start_ts,
            end_ts=end_ts,
            score=score,
        )
    return windows


def load_or_build_scenarios(df: pd.DataFrame) -> Dict[str, ScenarioWindow]:
    harness_dir = Path(config.HARNESS_DIR)
    harness_dir.mkdir(parents=True, exist_ok=True)
    scenario_file = harness_dir / "scenario_windows.json"
    dataset_sig = hash_dataframe(df)

    if scenario_file.exists():
        try:
            payload = json.loads(scenario_file.read_text(encoding="utf-8"))
            if payload.get("dataset_sig") == dataset_sig:
                return {
                    name: ScenarioWindow(**data)
                    for name, data in payload.get("windows", {}).items()
                }
        except Exception as exc:
            logger.warning(f"[EvalHarness] Failed to load scenarios: {exc}")

    windows = build_scenario_windows(
        df=df,
        window_bars=int(getattr(config, "EVAL_HARNESS_WINDOW_BARS", 400)),
        scenarios=getattr(config, "EVAL_HARNESS_SCENARIOS", {}),
    )
    payload = {
        "dataset_sig": dataset_sig,
        "created_at": time.time(),
        "window_bars": int(getattr(config, "EVAL_HARNESS_WINDOW_BARS", 400)),
        "windows": {k: asdict(v) for k, v in windows.items()},
    }
    scenario_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return windows


def load_policy_specs(max_policies: int) -> List[Any]:
    repo = LedgerRepo(config.LEDGER_DIR)
    records = [r for r in repo.load_records() if r.policy_spec]
    if not records:
        return []
    records.sort(key=lambda r: (r.cpcv_metrics or {}).get("eval_score", -9999.0), reverse=True)
    return [r.policy_spec for r in records[:max_policies]]


def run_eval_harness(df: pd.DataFrame, policies: Optional[List[Any]] = None) -> Dict[str, Any]:
    policies = policies or load_policy_specs(int(getattr(config, "EVAL_HARNESS_MAX_POLICIES", 10)))
    if not policies:
        raise ValueError("No policies available for eval harness")

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    scenarios = load_or_build_scenarios(df)
    detector = RegimeDetector()
    results_payload: Dict[str, Any] = {
        "timestamp": time.time(),
        "policy_count": len(policies),
        "scenarios": {},
    }

    for name, window in scenarios.items():
        window_df = df.iloc[window.start:window.end + 1]
        regime = detector.detect(window_df)
        stage_id = int(getattr(config, "CURRICULUM_CURRENT_STAGE", 1))
        results, diag = evaluate_stage(
            policies,
            window_df,
            "full",
            regime.label,
            n_jobs=1,
            stage_id=stage_id,
        )

        scenario_summary = []
        for res in results:
            if not res.best_sample:
                continue
            metrics = res.best_sample.metrics
            scenario_summary.append({
                "policy_id": res.policy_spec.spec_id,
                "score": res.score,
                "cycle_count": getattr(metrics.trades, "cycle_count", 0),
                "trade_count": metrics.trades.trade_count,
                "gate_pass": not metrics.is_rejected,
            })

        results_payload["scenarios"][name] = {
            "window": asdict(window),
            "regime": regime.label,
            "diagnostic": diag,
            "results": scenario_summary,
        }

    out_path = Path(config.HARNESS_DIR) / f"eval_harness_{int(time.time())}.json"
    out_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    return results_payload
