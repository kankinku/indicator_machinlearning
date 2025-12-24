
import json
import hashlib
from typing import Any, Dict

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
        "risk_budget": policy_spec.risk_budget,
        "execution": policy_spec.execution_assumption,
        "data_window": policy_spec.data_window,
    }
    return f"pol_{calculate_sha256(content)[:16]}"
