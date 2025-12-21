from __future__ import annotations

from src.contracts import PolicySpec
from src.l3_meta.state import MetaState


class RLAgent:
    """
    PPO/SAC 대체 스텁.
    실제 구현 시 상태-정책 매핑을 학습하도록 확장한다.
    """

    def propose(self, meta_state: MetaState, current_spec: PolicySpec) -> PolicySpec:
        # 현재는 제약/리스크 프리셋을 보수적으로 조정하는 예시만 제공
        tuned = current_spec.tuned_params.copy()
        risk_budget = current_spec.risk_budget.copy()
        if meta_state.performance_state.dd > 0.1:
            risk_budget["preset"] = "defensive"
            risk_budget["max_dd"] = min(risk_budget.get("max_dd", 0.15), 0.1)
        return PolicySpec(
            spec_id=current_spec.spec_id,
            template_id=current_spec.template_id,
            tuned_params=tuned,
            data_window=current_spec.data_window,
            risk_budget=risk_budget,
            execution_assumption=current_spec.execution_assumption,
        )

