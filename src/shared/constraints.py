from __future__ import annotations

from typing import Any, Dict, Optional

from src.config import config


def compute_entry_rate_constraints(stage_id: int, observed_act_rate: Optional[float] = None) -> Dict[str, Any]:
    stage_id = int(stage_id or getattr(config, "CURRICULUM_CURRENT_STAGE", 1))
    stage_cfg = getattr(config, "CURRICULUM_STAGES", {}).get(stage_id)
    if stage_cfg is None:
        stage_cfg = getattr(config, "CURRICULUM_STAGES", {}).get(1)

    min_entries = None
    if stage_cfg is not None:
        min_entries = getattr(stage_cfg, "min_entries_per_year", None)
        if min_entries is None:
            min_entries = getattr(stage_cfg, "min_trades_per_year", 0.0)
    min_entries = float(min_entries or 0.0)

    trading_days = float(getattr(config, "TRADING_DAYS_PER_YEAR", 252.0))
    steps_per_day = float(getattr(config, "DECISION_STEPS_PER_DAY", 1.0))

    opp_default = float(getattr(config, "ENTRY_EFFECTIVE_OPPORTUNITY_FACTOR_DEFAULT", 1.0))
    opp_map = getattr(config, "ENTRY_EFFECTIVE_OPPORTUNITY_FACTOR_BY_STAGE", {}) or {}
    opp_min = float(getattr(config, "ENTRY_EFFECTIVE_OPPORTUNITY_FACTOR_MIN", 0.01))
    opp_max = float(getattr(config, "ENTRY_EFFECTIVE_OPPORTUNITY_FACTOR_MAX", 1.0))
    opportunity = float(opp_map.get(stage_id, opp_default))
    opportunity = max(opp_min, min(opportunity, opp_max))

    denom = trading_days * steps_per_day * opportunity
    required_entry_rate = float(min_entries / denom) if denom > 0 else 0.0

    bounds = getattr(config, "ENTRY_HIT_RATE_BOUNDS_BY_STAGE", {}) or {}
    stage_bounds = bounds.get(stage_id) or {}
    bounds_max = float(stage_bounds.get("max", 1.0))

    safety_margin = float(getattr(config, "ENTRY_RATE_SAFETY_MARGIN", 1.2))
    min_scale = float(getattr(config, "ENTRY_RATE_MIN_SCALE", 0.8))
    recommended_min_rate = float(required_entry_rate * min_scale) if required_entry_rate > 0.0 else 0.0
    if required_entry_rate > 0.0:
        feasibility_margin = bounds_max / required_entry_rate
        feasibility_ok = bounds_max >= (required_entry_rate * safety_margin)
    else:
        feasibility_margin = float("inf")
        feasibility_ok = True

    recommended_opportunity = None
    recommended_required_entry_rate = None
    opportunity_factor_delta = None
    opportunity_factor_warn = False
    obs_value = None
    if observed_act_rate is not None:
        try:
            obs_value = float(observed_act_rate)
        except (TypeError, ValueError):
            obs_value = None
    if obs_value is not None:
        recommended_opportunity = max(opp_min, min(obs_value, opp_max))
        denom_recommended = trading_days * steps_per_day * recommended_opportunity
        recommended_required_entry_rate = float(min_entries / denom_recommended) if denom_recommended > 0 else 0.0
        opportunity_factor_delta = float(recommended_opportunity - opportunity)
        warn_delta = float(getattr(config, "ENTRY_EFFECTIVE_OPPORTUNITY_FACTOR_DELTA_WARN", 0.2))
        opportunity_factor_warn = abs(opportunity_factor_delta) >= warn_delta

    return {
        "required_entry_rate": required_entry_rate,
        "entry_rate_bounds_max": bounds_max,
        "feasibility_margin": feasibility_margin,
        "feasibility_ok": feasibility_ok,
        "recommended_min_rate": recommended_min_rate,
        "min_entries_per_year": min_entries,
        "trading_days_per_year": trading_days,
        "decision_steps_per_day": steps_per_day,
        "effective_opportunity_factor": opportunity,
        "recommended_opportunity_factor": recommended_opportunity,
        "recommended_required_entry_rate": recommended_required_entry_rate,
        "opportunity_factor_delta": opportunity_factor_delta,
        "opportunity_factor_warn": opportunity_factor_warn,
        "observed_act_rate_mean": obs_value,
        "safety_margin": safety_margin,
        "min_rate_scale": min_scale,
    }
