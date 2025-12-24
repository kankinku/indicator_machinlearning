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
from src.shared.metrics import (
    compute_trades_stats, compute_equity_stats, compute_window_metrics,
    WindowMetrics as SampleMetrics, # Alias for compatibility
    TradeStats, EquityStats
)

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


# [V16] SampleMetrics is now an alias for WindowMetrics from src.shared.metrics


# ========================================
# Utility Functions
# ========================================

def stable_hash(s: str) -> int:
    """
    문자열을 안정적인 해시 정수로 변환합니다.
    동일한 입력에 대해 항상 같은 값을 반환합니다.
    """
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)


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
    """
    평가 모듈의 고유 키를 생성합니다.
    동일한 설정에 대해 동일한 키를 반환하여 캐싱/비교에 사용됩니다.
    """
    parts = [
        template_id,
        regime_id,
        tp_dist_id,
        sl_dist_id,
        horizon_dist_id,
        entry_threshold_id,
        data_window_id,
        cost_model_id,
    ]
    return "|".join(parts)


def get_risk_distributions(
    template_id: str,
    regime_id: str,
    risk_profile_id: str = "DEFAULT",
    base_tp: Optional[float] = None,
    base_sl: Optional[float] = None,
    base_h: Optional[int] = None,
) -> Tuple[RiskDist, RiskDist, RiskDist]:
    """
    Risk Profile과 레짐에 맞는 TP/SL/Horizon 분포를 반환합니다.
    
    Returns:
        Tuple[RiskDist, RiskDist, RiskDist]: (tp_dist, sl_dist, horizon_dist)
    """
    profiles_list = get_default_risk_profiles()
    profiles = {p.profile_id: p for p in profiles_list}
    profile = profiles.get(risk_profile_id)
    
    # Base values with config defaults
    # If profile exists, use its ranges to bound the base values or as fallback
    # But here logic seems to rely on base_tp/sl/h passed from PolicySpec first.
    
    tp_low = base_tp * 0.5 if base_tp else config.RISK_K_UP_MIN * config.RISK_EST_DAILY_VOL
    tp_high = base_tp * 1.5 if base_tp else config.RISK_K_UP_MAX * config.RISK_EST_DAILY_VOL
    
    if profile:
        # If profile provided, use its range guidelines (optional logic enhancement)
        # For now, we stick to the original logic which seemed to mix them or ignore profile if base is present.
        # But to be safe and simple:
        pass
    
    sl_low = base_sl * 0.5 if base_sl else config.RISK_K_DOWN_MIN * config.RISK_EST_DAILY_VOL
    sl_high = base_sl * 1.5 if base_sl else config.RISK_K_DOWN_MAX * config.RISK_EST_DAILY_VOL
    
    h_low = max(config.RISK_HORIZON_MIN, int(base_h * 0.5) if base_h else config.RISK_HORIZON_MIN)
    h_high = min(config.RISK_HORIZON_MAX, int(base_h * 1.5) if base_h else config.RISK_HORIZON_MAX)
    
    # Adjust based on regime
    regime_lower = regime_id.lower() if regime_id else ""
    if "panic" in regime_lower:
        # 패닉 레짐: 타이트한 SL, 짧은 horizon
        sl_high = sl_high * 0.7
        h_high = min(h_high, 10)
    elif "bull" in regime_lower:
        # 강세 레짐: 더 넓은 TP
        tp_high = tp_high * 1.2
        
    tp_dist = RiskDist(
        dist_id=f"tp_{tp_low:.4f}_{tp_high:.4f}",
        low=tp_low,
        high=tp_high,
    )
    sl_dist = RiskDist(
        dist_id=f"sl_{sl_low:.4f}_{sl_high:.4f}",
        low=sl_low,
        high=sl_high,
    )
    h_dist = RiskDist(
        dist_id=f"h_{h_low}_{h_high}",
        low=float(h_low),
        high=float(h_high),
    )
    
    return tp_dist, sl_dist, h_dist


def sample_risk_params(
    tp_dist: RiskDist,
    sl_dist: RiskDist,
    h_dist: RiskDist,
    n_samples: int,
    seed: int,
) -> List[RiskSample]:
    """
    분포에서 n_samples 개의 리스크 파라미터 샘플을 생성합니다.
    
    Args:
        tp_dist: Take Profit 분포
        sl_dist: Stop Loss 분포
        h_dist: Horizon 분포
        n_samples: 생성할 샘플 수
        seed: 랜덤 시드
        
    Returns:
        List[RiskSample]: 리스크 파라미터 샘플 목록
    """
    rng = random.Random(seed)
    samples = []
    
    for i in range(n_samples):
        tp = rng.uniform(tp_dist.low, tp_dist.high)
        sl = rng.uniform(sl_dist.low, sl_dist.high)
        h = int(rng.uniform(h_dist.low, h_dist.high))
        
        sample_id = f"s{i}_{seed % 10000}"
        samples.append(RiskSample(
            sample_id=sample_id,
            tp_pct=tp,
            sl_pct=sl,
            horizon=h,
        ))
    
    return samples

def compute_sample_metrics(
    trade_returns: np.ndarray, 
    trade_count: int,
    bars_total: int,
    benchmark_roi_pct: float = 0.0,
    valid_trade_count: Optional[int] = None,
    full_returns: Optional[np.ndarray] = None,
    exposure_mask: Optional[np.ndarray] = None,
    complexity_score: float = 0.0
) -> SampleMetrics:
    """
    [V16] Unified Metric Extraction
    Redirects to src.shared.metrics.compute_window_metrics.
    """
    if full_returns is None:
        # If full_returns not provided, use trade_returns as a fallback (should avoid)
        full_returns = trade_returns if trade_returns.any() else np.zeros(bars_total)
        
    metrics = compute_window_metrics(
        window_id="SAMPLE",
        trade_returns=trade_returns,
        full_returns=full_returns,
        bars_total=bars_total,
        benchmark_roi_pct=benchmark_roi_pct,
        exposure_mask=exposure_mask,
        complexity_score=complexity_score
    )
    return metrics

def validate_sample(metrics: SampleMetrics) -> Tuple[bool, str]:
    """
    [V14] Validation Layer (Standardized Failure Codes)
    Uses period-normalized metrics (trades_per_year) and annualized excess return gates.
    """
    curr_stage = getattr(config, 'CURRICULUM_CURRENT_STAGE', 1)
    stage_cfg = config.CURRICULUM_STAGES.get(curr_stage, {})
    
    # 1. Minimum Activity (Annualized)
    min_tpy = getattr(stage_cfg, "min_trades_per_year", 2.0)
    if metrics.trades.trades_per_year < min_tpy:
        return False, f"FAIL_MIN_TRADES ({metrics.trades.trades_per_year:.1f} < {min_tpy})"

    # 2. Exposure Gate
    min_exp = getattr(stage_cfg, "min_exposure", 0.05)
    if metrics.equity.exposure_ratio < min_exp:
        return False, f"FAIL_LOW_EXPOSURE ({metrics.equity.exposure_ratio:.1%} < {min_exp:.1%})"

    # 3. Excess Return PA Gate (Stage Dependent)
    min_excess = getattr(stage_cfg, "min_excess_return_pa", 0.0)
    if metrics.equity.excess_return < min_excess:
        return False, f"FAIL_LOW_RETURN ({metrics.equity.excess_return:.1f}% < {min_excess}%)"

    # 4. Winrate Limits
    if metrics.trades.win_rate < config.VAL_WINRATE_MIN:
        return False, f"FAIL_WINRATE_LOW ({metrics.trades.win_rate:.1%} < {config.VAL_WINRATE_MIN:.1%})"
    if metrics.trades.win_rate > config.VAL_WINRATE_MAX:
        return False, f"FAIL_WINRATE_HIGH ({metrics.trades.win_rate:.1%} < {config.VAL_WINRATE_MAX:.1%})"

    # 5. MDD Gate
    max_mdd = getattr(stage_cfg, "max_mdd_pct", config.VAL_MAX_MDD_PCT)
    if metrics.equity.max_drawdown_pct > max_mdd:
        return False, f"FAIL_MDD_BREACH ({metrics.equity.max_drawdown_pct:.1f}% > {max_mdd}%)"

    # 6. Profit Factor Gate
    min_pf = getattr(stage_cfg, "min_profit_factor", 1.0)
    if metrics.trades.profit_factor < min_pf:
        return False, f"FAIL_PF ({metrics.trades.profit_factor:.2f} < {min_pf})"

    # 7. Anti-Luck (Lucky Strike)
    if metrics.trades.top1_share > config.ANTILUCK_TOP1_SHARE_MAX:
        return False, f"FAIL_LUCKY_STRIKE (T1: {metrics.trades.top1_share:.2f} > {config.ANTILUCK_TOP1_SHARE_MAX})"
        
    return True, "PASS"


def score_sample(metrics: WindowMetrics) -> float:
    """
    [V11] Evaluation Layer (Scoring)
    '잘 버는 전략' 우선순위로 점수 산출.
    
    점수 구조:
    1. 수익률 (CAGR) - 압도적 1순위
    2. 손익비(R:R) 보너스 - "크게 이기는" 전략 유도
    3. 리스크 패널티 (MDD, Vol)
    4. 거래 부족 패널티 ('보상'이 아닌 '감점')
    5. 승률 과도 패널티 (80% 이상은 의심)
    """
    score = 0.0
    
    # [1순위] 복리 성과 (Total Return or CAGR)
    score += metrics.equity.total_return_pct * config.SCORE_W_CAGR
    
    # [2순위] 손익비(R:R) 보너스 - "크게 이기는" 전략 유도
    target_rr = getattr(config, 'REWARD_TARGET_RR', 1.5)
    excellent_rr = getattr(config, 'REWARD_EXCELLENT_RR', 2.0)
    
    rr = metrics.trades.reward_risk
    if rr >= excellent_rr:
        # R:R >= 2.0: 높은 보너스
        rr_bonus = 50.0
    elif rr >= target_rr:
        # R:R >= 1.5: 보통 보너스
        rr_bonus = 25.0 + (rr - target_rr) / (excellent_rr - target_rr) * 25.0
    elif rr >= 1.0:
        # R:R 1.0~1.5: 작은 보너스
        rr_bonus = (rr - 1.0) / (target_rr - 1.0) * 25.0
    else:
        # R:R < 1.0: 패널티
        rr_bonus = -25.0 * (1.0 - rr)
    
    score += rr_bonus
    
    # [3순위] 리스크 패널티 (MDD, Vol)
    score -= metrics.equity.max_drawdown_pct * config.SCORE_W_MDD
    score -= metrics.equity.vol_pct * config.SCORE_W_VOL
    
    # [4순위] 거래 활동 - 점진적 감점 메커니즘
    # 0회 = 최대 패널티, 목표 = 패널티 0
    target_trades = config.SCORE_TARGET_TRADES
    tc = metrics.trades.trade_count
    if tc <= 0:
        # 무거래 = 최대 패널티
        score -= target_trades * config.SCORE_PENALTY_LOW_TRADE
    elif tc < target_trades:
        # 0 ~ 목표: 점진적으로 패널티 감소
        shortage = target_trades - tc
        score -= shortage * config.SCORE_PENALTY_LOW_TRADE
    # 목표 이상은 패널티 없음
        
    # [5순위] 승률 과적합 강력 견제 (80% 이상부터 패널티 시작!)
    wr = metrics.trades.win_rate
    if wr > config.SCORE_WINRATE_PENALTY_START:
        # 80% 초과분에 대해 강력 패널티
        excess_pct = (wr - config.SCORE_WINRATE_PENALTY_START) * 100.0
        score -= excess_pct * config.SCORE_PENALTY_HIGH_WR
        
    # [6순위] 승률 보너스 폐지 - 수익률이 높으면 보너스 없이도 상위권
    
    # 점수 하한선 적용
    return max(float(config.EVAL_SCORE_MIN), float(score))


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
    if metrics.trades.trade_count < config.VAL_MIN_TRADES:
        return True
    if metrics.equity.max_drawdown_pct > config.VAL_MAX_MDD_PCT:
        return True
    return False

