from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.config import config
from src.l3_meta.risk_profiles import get_default_risk_profiles
from src.shared.logger import get_logger

logger = get_logger("l1.evaluator")


@dataclass(frozen=True)
class RiskDist:
    dist_id: str
    low: float
    high: float
    dist_type: str = "uniform"


@dataclass(frozen=True)
class RiskSample:
    sample_id: str
    tp_pct: float
    sl_pct: float
    horizon: int


@dataclass
class SampleMetrics:
    total_return_pct: float
    mdd_pct: float
    reward_risk: float
    vol_pct: float
    trade_count: int
    win_rate: float


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def build_module_key(
    template_id: str,
    regime_id: str,
    tp_dist_id: str,
    sl_dist_id: str,
    horizon_dist_id: str,
    entry_threshold_id: str,
    data_window_id: str,
    cost_model_id: str,
) -> str:
    return "|".join([
        template_id,
        regime_id,
        tp_dist_id,
        sl_dist_id,
        horizon_dist_id,
        entry_threshold_id,
        data_window_id,
        cost_model_id,
    ])


def _clamp_range(low: float, high: float, floor_val: float) -> Tuple[float, float]:
    low_c = max(floor_val, low)
    high_c = max(low_c, high)
    return low_c, high_c


def get_risk_distributions(
    template_id: str,
    regime_id: str,
    risk_profile_id: str,
    base_tp: Optional[float],
    base_sl: Optional[float],
    base_h: Optional[int],
) -> Tuple[RiskDist, RiskDist, RiskDist]:
    profile_map = {p.profile_id: p for p in get_default_risk_profiles()}
    profile = profile_map.get(risk_profile_id) or profile_map.get("BALANCED")
    if profile is None:
        profile = list(profile_map.values())[0]

    est_vol = config.RISK_EST_DAILY_VOL
    floor_pct = config.EVAL_RISK_SCALE_FLOOR / 100.0
    jitter = config.EVAL_RISK_JITTER_PCT

    tp_low = profile.k_up_range[0] * est_vol
    tp_high = profile.k_up_range[1] * est_vol
    sl_low = profile.k_down_range[0] * est_vol
    sl_high = profile.k_down_range[1] * est_vol

    if base_tp is not None:
        tp_low = max(tp_low, base_tp * (1 - jitter))
        tp_high = min(tp_high, base_tp * (1 + jitter))
    if base_sl is not None:
        sl_low = max(sl_low, base_sl * (1 - jitter))
        sl_high = min(sl_high, base_sl * (1 + jitter))

    tp_low, tp_high = _clamp_range(tp_low, tp_high, floor_pct)
    sl_low, sl_high = _clamp_range(sl_low, sl_high, floor_pct)

    h_low, h_high = profile.horizon_range
    if base_h is not None:
        h_low = max(h_low, int(round(base_h * (1 - jitter))))
        h_high = min(h_high, int(round(base_h * (1 + jitter))))
        if h_low > h_high:
            h_low = h_high

    tp_dist_id = f"tp_{template_id}_{regime_id}_{risk_profile_id}"
    sl_dist_id = f"sl_{template_id}_{regime_id}_{risk_profile_id}"
    h_dist_id = f"h_{template_id}_{regime_id}_{risk_profile_id}"

    tp_dist = RiskDist(dist_id=tp_dist_id, low=tp_low, high=tp_high, dist_type="uniform")
    sl_dist = RiskDist(dist_id=sl_dist_id, low=sl_low, high=sl_high, dist_type="uniform")
    h_dist = RiskDist(dist_id=h_dist_id, low=float(h_low), high=float(h_high), dist_type="uniform")
    return tp_dist, sl_dist, h_dist


def _sample_truncnorm(rng: random.Random, low: float, high: float) -> float:
    if high <= low:
        return low
    mean = 0.5 * (low + high)
    std = (high - low) / 6.0 if high > low else 1e-6
    for _ in range(50):
        val = rng.gauss(mean, std)
        if low <= val <= high:
            return val
    return min(max(mean, low), high)


def sample_risk_params(
    tp_dist: RiskDist,
    sl_dist: RiskDist,
    h_dist: RiskDist,
    sample_count: int,
    seed: int,
) -> List[RiskSample]:
    rng = random.Random(seed)
    samples: List[RiskSample] = []
    for i in range(sample_count):
        if tp_dist.dist_type == "truncnorm":
            tp = _sample_truncnorm(rng, tp_dist.low, tp_dist.high)
        else:
            tp = rng.uniform(tp_dist.low, tp_dist.high)

        if sl_dist.dist_type == "truncnorm":
            sl = _sample_truncnorm(rng, sl_dist.low, sl_dist.high)
        else:
            sl = rng.uniform(sl_dist.low, sl_dist.high)

        if h_dist.dist_type == "truncnorm":
            h_val = _sample_truncnorm(rng, h_dist.low, h_dist.high)
        else:
            h_val = rng.uniform(h_dist.low, h_dist.high)

        horizon = max(1, int(round(h_val)))
        sample_id = f"s{i:03d}"
        samples.append(RiskSample(sample_id=sample_id, tp_pct=tp, sl_pct=sl, horizon=horizon))
    return samples


def compute_trade_returns(
    results_df,
    tp_pct: float,
    sl_pct: float,
    cost_bps: float,
) -> np.ndarray:
    if results_df is None or results_df.empty:
        return np.array([], dtype=float)

    cost_pct = (2 * cost_bps) / 10000.0
    returns: List[float] = []
    for _, row in results_df.iterrows():
        pred = row.get("pred", 0)
        if pred == 0:
            returns.append(0.0)
            continue

        actual = row.get("actual", 0)
        scale = row.get("scale", 1.0)
        if actual == 0:
            base_ret = 0.0
        elif pred == actual:
            base_ret = tp_pct
        else:
            base_ret = -sl_pct

        trade_ret = (base_ret * scale) - (cost_pct * scale)
        returns.append(trade_ret)
    return np.array(returns, dtype=float)


def compute_sample_metrics(trade_returns: np.ndarray, trade_count: int) -> SampleMetrics:
    if trade_returns.size == 0:
        return SampleMetrics(
            total_return_pct=0.0,
            mdd_pct=0.0,
            reward_risk=0.0,
            vol_pct=0.0,
            trade_count=0,
            win_rate=0.0,
        )

    equity = np.cumprod(1.0 + trade_returns)
    total_return_pct = (equity[-1] - 1.0) * 100.0

    peak = np.maximum.accumulate(equity)
    drawdown = (equity / peak) - 1.0
    mdd_pct = abs(np.min(drawdown)) * 100.0

    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    if wins.size > 0 and losses.size > 0:
        reward_risk = np.median(wins) / abs(np.median(losses))
    else:
        reward_risk = 0.0

    vol_pct = float(np.std(trade_returns)) * 100.0
    win_rate = float((trade_returns > 0).sum() / max(1, (trade_returns != 0).sum()))

    return SampleMetrics(
        total_return_pct=float(total_return_pct),
        mdd_pct=float(mdd_pct),
        reward_risk=float(reward_risk),
        vol_pct=float(vol_pct),
        trade_count=int(trade_count),
        win_rate=win_rate,
    )


def score_sample(metrics: SampleMetrics) -> float:
    trade_score = min(metrics.trade_count / float(config.EVAL_TRADE_TARGET), 1.0)
    score = (
        (config.EVAL_SCORE_W_RETURN * metrics.total_return_pct)
        + (config.EVAL_SCORE_W_RR * metrics.reward_risk)
        - (config.EVAL_SCORE_W_MDD * metrics.mdd_pct)
        - (config.EVAL_SCORE_W_VOL * metrics.vol_pct)
        + (config.EVAL_SCORE_W_TRADE * trade_score)
        + (config.EVAL_SCORE_W_WINRATE * metrics.win_rate * 100.0)
    )
    return float(score)


def aggregate_sample_scores(sample_scores: List[float], violations: List[bool]) -> Dict[str, float]:
    if not sample_scores:
        return {
            "median_score": 0.0,
            "p10_score": 0.0,
            "std_score": 0.0,
            "violation_rate": 0.0,
            "score": 0.0,
        }

    scores = np.array(sample_scores, dtype=float)
    lower_q = config.EVAL_LOWER_QUANTILE
    median_score = float(np.median(scores))
    p10_score = float(np.quantile(scores, lower_q))
    std_score = float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0
    violation_rate = float(sum(violations) / max(1, len(violations)))

    score = (
        median_score
        + (config.EVAL_SCORE_W_RETURN_Q * p10_score)
        - (config.EVAL_SCORE_W_VOL * std_score)
        - (config.EVAL_SCORE_W_VIOLATION * violation_rate)
    )

    return {
        "median_score": median_score,
        "p10_score": p10_score,
        "std_score": std_score,
        "violation_rate": violation_rate,
        "score": score,
    }


def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    arr = np.array(scores, dtype=float)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if std <= 1e-12:
        return [0.0 for _ in scores]
    mean = float(np.mean(arr))
    return [float((val - mean) / std) for val in scores]


def is_violation(metrics: SampleMetrics) -> bool:
    if metrics.trade_count < config.EVAL_MIN_TRADES:
        return True
    if metrics.mdd_pct > config.EVAL_MAX_DD_PCT:
        return True
    return False

