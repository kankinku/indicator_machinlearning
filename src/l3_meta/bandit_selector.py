from __future__ import annotations

from typing import Dict

from src.contracts import PolicySpec
from src.l3_meta.state import MetaState
from src.templates.registry import TemplateRegistry


def _default_params(template) -> Dict[str, object]:
    params: Dict[str, object] = {}
    for spec in template.tunable_params_schema:
        params[spec.name] = spec.default
    return params


def select_policy_spec(meta_state: MetaState, registry: TemplateRegistry, spec_id: str) -> PolicySpec:
    """
    간단한 컨텍스트 밴딧 대체 스텁:
    - PBO가 높거나 DD가 크면 보수적 템플릿(T08) 우선
    - 그렇지 않으면 기본 템플릿(T01) 사용
    """
    if meta_state.performance_state.pbo > 0.4 or meta_state.performance_state.dd > 0.1:
        template_id = "T08"
    else:
        template_id = "T01"

    template = registry.get_template(template_id)
    tuned_params = template.validate_params(_default_params(template))
    risk_preset = "conservative" if template_id == "T08" else "balanced"

    return PolicySpec(
        spec_id=spec_id,
        template_id=template_id,
        tuned_params=tuned_params,
        data_window={"train": "recent", "valid": "recent"},
        risk_budget={"preset": risk_preset, "max_dd": template.base_constraints.get("max_dd", 0.15)},
        execution_assumption={"fee_bps": 1, "slippage_bps": 2, "fill": "aggressive"},
    )

