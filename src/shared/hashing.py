
import json
import hashlib
from typing import Any, Dict

import pandas as pd
from src.config import config

def get_canonical_json(data: Any) -> str:
    """Returns a canonical JSON string (keys sorted, floats rounded)."""
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [_sanitize(x) for x in obj]
        elif isinstance(obj, float):
            return round(obj, 8)
        return obj

    return json.dumps(_sanitize(data), separators=(",", ":"))

def calculate_sha256(data: Any) -> str:
    """Calculates SHA256 hash of canonical JSON representation."""
    canonical = get_canonical_json(data)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def generate_policy_id(policy_spec: Any) -> str:
    """
    Generates a deterministic policy_id based on:
    - genome (feature_genome)
    - decision_rules
    - eval_config (risk_budget, execution_assumption, data_window)
    """
    # Keys to include for structural identity
    content = {
        "genome": policy_spec.feature_genome,
        "rules": policy_spec.decision_rules,
        "logic_trees": getattr(policy_spec, 'logic_trees', {}),
        "risk_budget": policy_spec.risk_budget,
        "execution": policy_spec.execution_assumption,
        "data_window": policy_spec.data_window,
    }
    return f"pol_{calculate_sha256(content)[:16]}"


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Stable hash for DataFrame contents + index/columns.
    """
    try:
        content_hash = pd.util.hash_pandas_object(df, index=True).sum()
        col_hash = hash(tuple(df.columns.tolist()))
        shape_hash = hash(df.shape)
        payload = f"{content_hash}_{col_hash}_{shape_hash}"
    except Exception:
        payload = str(df.values.tobytes())
    return calculate_sha256(payload)


def get_eval_config_signature() -> str:
    """
    Hash of evaluation-relevant config to enforce deterministic caching.
    """
    eval_cfg = {
        "EVAL_RISK_SAMPLES_FAST": config.EVAL_RISK_SAMPLES_FAST,
        "EVAL_RISK_SAMPLES_REDUCED": config.EVAL_RISK_SAMPLES_REDUCED,
        "EVAL_RISK_SAMPLES_FULL": config.EVAL_RISK_SAMPLES_FULL,
        "EVAL_WINDOW_COUNT_FAST": config.EVAL_WINDOW_COUNT_FAST,
        "EVAL_WINDOW_COUNT_REDUCED": config.EVAL_WINDOW_COUNT_REDUCED,
        "EVAL_WINDOW_COUNT_FULL": config.EVAL_WINDOW_COUNT_FULL,
        "EVAL_MIN_WINDOW_BARS": config.EVAL_MIN_WINDOW_BARS,
        "WF_GATE_ENABLED": getattr(config, "WF_GATE_ENABLED", True),
        "WF_SPLITS_STAGE1": getattr(config, "WF_SPLITS_STAGE1", 2),
        "WF_SPLITS_STAGE2": getattr(config, "WF_SPLITS_STAGE2", 3),
        "WF_SPLITS_STAGE3": getattr(config, "WF_SPLITS_STAGE3", 4),
        "EVAL_CAPS_BY_STAGE": getattr(config, "EVAL_CAPS_BY_STAGE", {}),
        "EVAL_SCORE_MIN": config.EVAL_SCORE_MIN,
        "EVAL_CACHE_VERSION": getattr(config, "EVAL_CACHE_VERSION", 0),
    }
    return calculate_sha256(eval_cfg)[:16]
