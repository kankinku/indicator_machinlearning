from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def get_risk_unit(risk_budget: Optional[dict], default_pct: float = 0.01) -> float:
    """
    Returns the per-trade risk unit as a fraction (e.g. 0.02 for 2%).
    Falls back to default_pct if stop_loss is unavailable.
    """
    if not risk_budget:
        return default_pct

    stop_loss = risk_budget.get("stop_loss")
    if stop_loss is not None:
        try:
            return float(stop_loss)
        except (TypeError, ValueError):
            return default_pct

    k_down = risk_budget.get("k_down")
    if k_down is None:
        return default_pct

    try:
        # Match MetaAgent stop_loss approximation.
        stop_loss = float(k_down) * 0.015
    except (TypeError, ValueError):
        return default_pct

    # Clamp to a reasonable range to avoid pathological compounding.
    return max(0.001, min(0.2, stop_loss))


def get_risk_reward_ratio(risk_budget: Optional[dict]) -> Optional[float]:
    if not risk_budget:
        return None
    ratio = risk_budget.get("risk_reward_ratio")
    if ratio is not None:
        try:
            return float(ratio)
        except (TypeError, ValueError):
            return None

    k_up = risk_budget.get("k_up")
    k_down = risk_budget.get("k_down")
    try:
        if k_up is not None and k_down is not None and float(k_down) != 0:
            return float(k_up) / float(k_down)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def resolve_results_csv(exp_id: str, model_artifact_ref: str, ledger_dir: Path) -> Optional[Path]:
    if model_artifact_ref:
        artifact_path = Path(model_artifact_ref)
        candidate = artifact_path.parent / f"{artifact_path.stem}_results.csv"
        if candidate.exists():
            return candidate

    fallback = Path(ledger_dir) / "artifacts" / f"{exp_id}_results.csv"
    if fallback.exists():
        return fallback

    return None


def compute_compounded_return_pct(
    exp_id: str,
    model_artifact_ref: str,
    ledger_dir: Path,
    risk_budget: Optional[dict],
    default_risk_unit: float = 0.01,
) -> Optional[float]:
    results_path = resolve_results_csv(exp_id, model_artifact_ref, ledger_dir)
    if not results_path:
        return None

    df = pd.read_csv(results_path)
    if "net_pnl" not in df.columns:
        return None

    risk_unit = get_risk_unit(risk_budget, default_pct=default_risk_unit)
    returns = pd.to_numeric(df["net_pnl"], errors="coerce").fillna(0.0)
    pct_returns = returns * risk_unit
    pct_returns = pct_returns.clip(lower=-0.95)

    equity = (1.0 + pct_returns).cumprod()
    if equity.empty:
        return None
    return float((equity.iloc[-1] - 1.0) * 100.0)


def compute_equity_curve_pct(
    exp_id: str,
    model_artifact_ref: str,
    ledger_dir: Path,
    risk_budget: Optional[dict],
    default_risk_unit: float = 0.01,
) -> Optional[pd.Series]:
    results_path = resolve_results_csv(exp_id, model_artifact_ref, ledger_dir)
    if not results_path:
        return None

    df = pd.read_csv(results_path)
    if "net_pnl" not in df.columns:
        return None

    risk_unit = get_risk_unit(risk_budget, default_pct=default_risk_unit)
    returns = pd.to_numeric(df["net_pnl"], errors="coerce").fillna(0.0)
    pct_returns = returns * risk_unit
    pct_returns = pct_returns.clip(lower=-0.95)

    equity_pct = (1.0 + pct_returns).cumprod()
    if equity_pct.empty:
        return None
    return (equity_pct - 1.0) * 100.0
