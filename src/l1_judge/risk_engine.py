from __future__ import annotations

from typing import Dict


def assess_risk(forecast_risk: Dict[str, float], budget: Dict[str, float]) -> Dict[str, float]:
    """
    예상 변동성과 예산을 비교해 간단한 리스크 리포트를 만든다.
    """
    sigma_hat = forecast_risk.get("sigma_hat", 0.0)
    max_dd = budget.get("max_dd", 0.15)
    leverage_cap = budget.get("leverage_cap", 1.0)
    return {
        "sigma_hat": sigma_hat,
        "max_dd": max_dd,
        "leverage_cap": leverage_cap,
        "risk_ok": sigma_hat < max_dd,
    }

