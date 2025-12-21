from __future__ import annotations

from typing import Dict, Iterable, List

from src.contracts import FixSuggestion


def generate_fix_suggestion(reason_codes: Iterable[str]) -> FixSuggestion:
    reasons = list(reason_codes)
    param_changes: Dict[str, object] = {}
    constraints: Dict[str, object] = {}
    template_switch = None

    if "CALIBRATION_BAD" in reasons:
        param_changes["entry_threshold"] = "+0.02"
        constraints["calibration"] = "switch_calibrator"
    if "TURNOVER_TOO_HIGH" in reasons:
        constraints["min_holding"] = "increase"
        param_changes["entry_threshold"] = "+0.03"
    if "DD_LIMIT_BREACH" in reasons:
        constraints["max_dd"] = "tighten"
        param_changes["k"] = "-0.1"
    if "REGIME_DEPENDENT" in reasons:
        template_switch = "T08"

    return FixSuggestion(
        suggested_param_changes=param_changes,
        suggested_constraints=constraints,
        suggested_template_switch=template_switch,
    )

