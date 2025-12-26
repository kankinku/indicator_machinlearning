"""
EAGL - Economic Alpha Guidance Layer [vAlpha+]
전략의 '경제적 가치'를 평가하고 탐색 우선순위를 가이드합니다.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import time
from src.config import config
from src.contracts import PolicySpec, EvaluationResult
from src.shared.logger import get_logger

logger = get_logger("l3.eagl")

class EAGLEngine:
    def __init__(self):
        self.enabled = config.EAGL_ENABLED
        self.tau = config.AOS_TAU
        
    def calculate_aos(self, result: EvaluationResult) -> float:
        """
        Alpha Opportunity Score (AOS) 계산 [0, 1]
        수익성, 비용 생존력, 거래 빈도, 일관성을 종합 평가.
        """
        if not result.best_sample:
            return 0.0
            
        metrics = result.best_sample.metrics
        
        # 1. Return/Cost Factor (수익 대비 비용 효율)
        # Net Return이 비용(BPS) 대비 얼마나 높은지
        total_ret = metrics.equity.total_return_pct
        cost_bps = float(result.policy_spec.execution_assumption.get("cost_bps", 5))
        trade_count = metrics.trades.trade_count
        
        if trade_count <= 0:
            return 0.0
            
        # 예상 총 비용 (대략적)
        est_total_cost = (trade_count * cost_bps * 0.01) # bps -> %
        return_cost_ratio = total_ret / (est_total_cost + 0.1)
        # Sigmoid-like mapping to [0, 1]
        f_ret_cost = np.tanh(max(0, return_cost_ratio) / 5.0)
        
        # 2. Frequency Factor (경제적 유의미성 빈도)
        # 너무 적은 거래는 통계적 신뢰도가 낮음
        tpy = metrics.trades.trades_per_year
        f_freq = np.tanh(tpy / 100.0) # 100회/년 기준 포화
        
        # 3. Consistency Factor (일관성)
        # Walk-forward alpha 일관성 기반
        alphas = [w.avg_alpha for w in result.window_results]
        if not alphas:
            f_cons = 0.0
        else:
            p_consistent = sum(1 for a in alphas if a > 0) / len(alphas)
            f_cons = p_consistent
            
        # Weighted Final AOS
        aos = (
            config.AOS_WEIGHT_RETURN_COST * f_ret_cost +
            config.AOS_WEIGHT_FREQUENCY * f_freq +
            config.AOS_WEIGHT_CONSISTENCY * f_cons
        )
        
        return float(np.clip(aos, 0.0, 1.0))

    def evaluate_viability(self, result: EvaluationResult) -> Tuple[bool, str]:
        """경제적 생존 가능성(Viability) 판정"""
        if not result.best_sample:
            return False, "NO_BEST_SAMPLE"
            
        metrics = result.best_sample.metrics
        cost_bps = float(result.policy_spec.execution_assumption.get("cost_bps", 5))
        
        # [Rule] Net Return + Alpha가 예상 비용의 2배 미만이면 비경제적
        est_total_cost = (metrics.trades.trade_count * cost_bps * 0.01)
        if metrics.equity.total_return_pct < (est_total_cost * 1.5):
            return False, f"LOW_NET_PnL (Cost: {est_total_cost:.2f}%)"
            
        # [Rule] 년간 거래 횟수가 너무 적음 (고정 비용 및 슬리피지 감당 불가)
        if metrics.trades.trades_per_year < 15:
            return False, "LOW_FREQUENCY"
            
        return True, "VIABLE"

    def reallocate_budget(self, policies: List[PolicySpec], regime: Optional[Any] = None) -> List[float]:
        """
        Exploration Budget Reallocator (EBR) [Alpha-Power V1]
        AOS + Market Context(Session, Vol) 기반으로 탐색 가중치 계산.
        """
        if not policies:
            return []
            
        aos_scores = np.array([p.aos_score for p in policies])
        
        # Softmax base
        exp_aos = np.exp(aos_scores / self.tau)
        base_weights = exp_aos / np.sum(exp_aos)
        
        # [Alpha-Power V1] Context Multipliers
        final_weights = base_weights.copy()
        if regime:
            # 1. Session Multiplier (e.g. London/NY favored for exploration)
            sess_mult = self._get_session_multiplier(regime.session_id)
            # 2. Vol Squeeze Multiplier (Low vol -> High budget for discovery)
            vol_mult = self._get_vol_multiplier(regime.vol_level)
            
            final_weights *= (sess_mult * vol_mult)
            final_weights /= np.sum(final_weights) # Re-normalize
            
        return final_weights.tolist()

    def _get_session_multiplier(self, session_id: int) -> float:
        # NY(2), London(1) > Asia(0), Wrap(3)
        mults = {0: 0.8, 1: 1.2, 2: 1.2, 3: 0.5}
        return mults.get(session_id, 1.0)
        
    def _get_vol_multiplier(self, vol_level: float) -> float:
        # Squeeze (Low Vol Relative) -> Higher discovery budget
        if vol_level < 0.7: return 1.5
        if vol_level > 2.0: return 0.5 # Extreme high vol -> Safety first, reduce exploration
        return 1.0

    def should_discount_trust(self, regime: Any) -> Tuple[bool, float]:
        """
        [Alpha-Power V1] De-sync Detection
        Correlation (Close vs Volume) 붕괴 시 Trust Factor 할인.
        """
        if abs(regime.corr_score) < 0.2: # Correlation breakdown
            return True, 0.5 # 50% discount on trust
        return False, 1.0


    def update_policy_status(self, policy: PolicySpec, success: bool):
        """Conditional Revival Mechanism (CRM) 상태 업데이트"""
        if success:
            policy.failure_count = 0
            policy.status = "active"
        else:
            policy.failure_count += 1
            if policy.failure_count >= config.CRM_DORMANT_THRESHOLD:
                policy.status = "dormant"
                logger.debug(f"Policy {policy.spec_id} marked as DORMANT.")

    def should_revive(self, policy: PolicySpec, current_market_context: Dict[str, Any]) -> bool:
        """휴면 전략의 부활 조건 체크"""
        if policy.status != "dormant":
            return False
            
        # [Example] 비용 구조가 변했거나 특정 레짐이 돌아왔을 때 부활
        # 여기서는 단순화하여 일정 시간(24시간)이 지나면 재시도 기회를 줌
        if (time.time() - policy.created_at) > 86400:
            return True
            
        return False

_eagl_engine = None
def get_eagl_engine() -> EAGLEngine:
    global _eagl_engine
    if _eagl_engine is None:
        _eagl_engine = EAGLEngine()
    return _eagl_engine
