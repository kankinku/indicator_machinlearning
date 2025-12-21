from __future__ import annotations

from typing import Dict

from src.contracts import StrategyTemplate, ValidationError


def light_tune(template: StrategyTemplate, params: Dict[str, object], hints: Dict[str, object]) -> Dict[str, object]:
    """
    간단한 ES 스타일 튜닝 스텁.
    - hints에 따라 특정 파라미터를 상향/하향 조정한다.
    - 스키마 검증을 통과하지 못하면 ValidationError를 던진다.
    """
    tuned = params.copy()
    for key, value in hints.items():
        if key not in tuned:
            tuned[key] = value
            continue
        current = tuned[key]
        try:
            new_value = current + value  # value는 delta로 가정
        except TypeError:
            new_value = value
        tuned[key] = new_value

    return template.validate_params(tuned)

