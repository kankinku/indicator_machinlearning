from __future__ import annotations

from typing import Iterable, List

from src.contracts import PolicySpec
from src.orchestration.run_experiment import run_experiment
from src.templates.registry import TemplateRegistry


def walkforward(
    registry: TemplateRegistry,
    policy_specs: Iterable[PolicySpec],
    price_windows: List[List[float]],
) -> List[object]:
    """
    단순 워크포워드 스텁: 각 윈도우마다 주어진 PolicySpec을 순서대로 사용한다.
    """
    results = []
    for spec, window in zip(policy_specs, price_windows):
        results.append(run_experiment(registry, spec, window))
    return results

