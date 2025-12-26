"""
Reward Shaper - 다면적 보상 함수 (Final Design)

[설계도 V11] 
목표: "거래를 회피하지 않는 + 유리한 국면에서만 진입 + 손익비로 크게 이기는" 전략

핵심 철학:
- Reward는 "방향을 가르치는 도구" (순위 결정 ❌ / 행동 교정 ⭕)
- "안전함" 보상 ❌
- "시도 + 질 좋은 시도" 보상 ⭕

보상 구성:
1. 거래 활동성 보상 (방향성 필터 포함)
2. 손익비(R:R) 보상
3. 상위 트레이드 기여도 보상
4. 수익률 보상 (주력)
5. 승률 처리 (보조 - 과도 패널티)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from src.config import config
from src.shared.logger import get_logger

logger = get_logger("l3.reward_shaper")


@dataclass
class RewardBreakdown:
    """보상의 세부 구성 요소."""
    total: float
    
    # 주력 성분
    return_component: float          # 수익률 보상 (Alpha)
    rr_component: float              # 손익비 보상
    top_trades_component: float      # 상위 트레이드 기여도 보상
    
    # 보조 성분
    regime_trade_component: float    # 레짐 일치 거래 보상
    trades_component: float          # 거래 활동성 보상
    
    # 패널티 성분
    mdd_penalty: float               # MDD 패널티
    winrate_penalty: float           # 승률 과적합 패널티
    
    # [vAlpha+] 경제적 알파 성분
    repeatability_component: float = 0.0
    cost_survival_component: float = 0.0
    stability_component: float = 0.0
    
    # 상태 정보
    is_rejected: bool = False                # Hard Gate 통과 여부
    rejection_reason: Optional[str] = None  # 통과 실패 사유
    raw_metrics: Dict[str, float] = field(default_factory=dict)    # 원본 지표

    failure_codes: List[str] = field(default_factory=list)
    distance_score: float = 0.0
    replay_tag: str = "PASS"
    alpha: float = 0.0              # [V11.2] 초과 수익률 (연간화)





class RewardShaper:
    """
    다면적 보상 함수 - Final Design.
    
    핵심 원칙:
    1. Selection Pressure: Gate 미달 시 즉시 REJECTED 및 학습 가치 0 (최저점)
    2. Alpha = Strategy Return - Benchmark CAGR (벤치 초과 수익 기반)
    3. Regime Gating: 레짐 불일치 거래는 활동성으로 인정하지 않음
    4. Quality over Quantity: 손익비(R:R) 및 상위 트레이드 기여도 강조
    """
    
    def __init__(
        self,
        w_return: float = None,
        w_rr: float = None,
        w_top_trades: float = None,
        w_regime_trade: float = None,
        w_trades: float = None,
        w_mdd: float = None,
    ):
        # 가중치 설정 (Config 우선)
        self.w_return = w_return or getattr(config, 'REWARD_W_RETURN', 2.0)
        self.w_rr = w_rr or getattr(config, 'REWARD_W_RR', 1.0)
        self.w_top_trades = w_top_trades or getattr(config, 'REWARD_W_TOP_TRADES', 0.5)
        self.w_regime_trade = w_regime_trade or getattr(config, 'REWARD_W_REGIME_TRADE', 0.3)
        self.w_trades = w_trades or getattr(config, 'REWARD_W_TRADES', 0.6)
        self.w_mdd = w_mdd or getattr(config, 'REWARD_W_MDD', 0.3)
        
        # 스케일 및 임계값
        self.return_scale = config.REWARD_RETURN_SCALE
        self.mdd_threshold = config.REWARD_MDD_THRESHOLD
        self.min_trades = config.REWARD_MIN_TRADES
        self.target_trades = config.REWARD_TARGET_TRADES
        self.max_trades = config.REWARD_MAX_TRADES
        
        # 손익비 목표
        self.target_rr = getattr(config, 'REWARD_TARGET_RR', 1.5)
        self.excellent_rr = getattr(config, 'REWARD_EXCELLENT_RR', 2.0)
        
        # 상위 트레이드 기여도 목표
        self.top_trade_pct = getattr(config, 'REWARD_TOP_TRADE_PCT', 0.2)  # 상위 20%
        self.top_trade_contrib_target = getattr(config, 'REWARD_TOP_TRADE_CONTRIB_TARGET', 0.6)  # 60% 기여
        
        # 승률 임계값
        self.winrate_penalty_start = getattr(config, 'SCORE_WINRATE_PENALTY_START', 0.80)
        self.winrate_ideal_min = getattr(config, 'SCORE_TARGET_WINRATE_MIN', 0.40)
        self.winrate_ideal_max = getattr(config, 'SCORE_TARGET_WINRATE_MAX', 0.70)
    def compute(self, metrics: Dict[str, float]) -> float:
        """평가 지표에서 복합 보상을 계산합니다."""
        breakdown = self.compute_breakdown(metrics)
        return breakdown.total
    
    def compute_breakdown(self, metrics: Dict[str, float]) -> RewardBreakdown:
        """
        보상의 세부 구성을 계산합니다.
        
        필수 입력 (Backtest 결과에서):
        - total_return_pct: Backtest 총 수익률 (%)
        - mdd_pct: Backtest MDD (%)
        - win_rate: Backtest 승률 (0~1)
        - n_trades: Backtest 거래 횟수
        
        선택 입력:
        - reward_risk: 평균 손익비 (R:R)
        - trade_returns: 개별 트레이드 수익률 리스트 (상위 기여도 계산용)
        - regime_aligned_trades: 레짐 일치 거래 수
        """
        # [0] 커리큘럼 기반 Stage 정보 획득
        current_stage = getattr(config, 'CURRICULUM_CURRENT_STAGE', 1)
        stages = getattr(config, 'CURRICULUM_STAGES', {})
        # Ensure we get a StageSpec object, not a dict, to support getattr() downstream
        stage_cfg = stages.get(current_stage)
        if stage_cfg is None:
            stage_cfg = stages.get(1)  # Fallback to Stage 1
            if stage_cfg is None:
                # Extreme fallback if config is broken
                from src.shared.stage_schema import StageSpec
                stage_cfg = StageSpec(1, "Fallback", 15.0, 0.0, 6.0, 40.0, 1.0)
        
        stage_benchmark = getattr(stage_cfg, "target_return_pct", 50.0)
        stage_min_trades = getattr(stage_cfg, "min_trades_per_year", 20.0)
        
        # [1] 데이터 추출 (shared/metrics에서 계산된 값들 우선)
        total_return_pct = metrics.get("total_return_pct", 0.0)
        mdd_pct = abs(metrics.get("mdd_pct", 0.0))
        n_trades = metrics.get("n_trades", metrics.get("trade_count", 0))
        win_rate = metrics.get("win_rate", 0.5)
        reward_risk = metrics.get("reward_risk", 1.0)
        
        # Alpha (excess_return) 가 있으면 사용, 없으면 직접 계산 자제 (SSOT 원칙)
        alpha = metrics.get("excess_return", 0.0)
        if "excess_return" not in metrics and "total_return_pct" in metrics:
            days = metrics.get("oos_bars", 252)
            cagr = ((1 + total_return_pct/100) ** (252/max(1, days)) - 1) * 100
            bench_cagr = getattr(config, 'REWARD_BENCHMARK_RETURN', 15.0)
            alpha = cagr - bench_cagr

        # [V12.3] Auto-adjust Return Weight by Stage
        if current_stage == 1: self.w_return = 1.0
        elif current_stage == 2: self.w_return = 2.0
        else: self.w_return = 3.0

        # [2] Validation Failures (Decomposed)
        # ================================================
        from src.l1_judge.evaluator import collect_validation_failures_from_dict
        failures = collect_validation_failures_from_dict(metrics)
        hard_failures = [f for f in failures if f.is_hard]
        is_rejected = len(hard_failures) > 0
        rejection_reason = hard_failures[0].code if hard_failures else None

        reason_penalties = getattr(config, "REJECT_REASON_PENALTIES", {})
        distance_weights = getattr(config, "REJECT_DISTANCE_PENALTY_WEIGHTS", {})
        failure_codes = [f.code for f in failures]
        distance_score = float(sum(f.distance for f in failures))
        reason_penalty_total = float(sum(reason_penalties.get(code, 0.0) for code in failure_codes))
        distance_penalty_total = float(sum(distance_weights.get(f.code, 0.0) * f.distance for f in failures))

        # [3] REJECTED 처리 - 분해된 패널티 적용
        # ================================================
        if is_rejected:
            stage_base = getattr(stage_cfg, "reject_base_penalty", None)
            if stage_base is not None:
                base_penalty = float(stage_base)
            else:
                base_map = getattr(config, "REJECT_BASE_PENALTY_BY_STAGE", {})
                base_penalty = float(base_map.get(current_stage, getattr(config, "RL_REJECT_SCORE", -50.0)))
            total = base_penalty + reason_penalty_total + distance_penalty_total
            total = max(total, float(getattr(config, "REJECT_SCORE_FLOOR", -100.0)))

            hard_dist = float(sum(f.distance for f in hard_failures))
            near_max_failures = int(getattr(config, "REJECT_NEAR_PASS_MAX_FAILURES", 2))
            near_max_distance = float(getattr(config, "REJECT_NEAR_PASS_MAX_DISTANCE", 0.35))
            replay_tag = "NEAR_PASS" if (len(hard_failures) <= near_max_failures and hard_dist <= near_max_distance) else "HARD_FAIL"

            return RewardBreakdown(
                total=float(total),
                return_component=-1.0,
                rr_component=-1.0,
                top_trades_component=-1.0,
                regime_trade_component=-1.0,
                trades_component=-1.0,
                mdd_penalty=-1.0,
                winrate_penalty=-1.0,
                is_rejected=True,
                rejection_reason=rejection_reason,
                raw_metrics=metrics,
                failure_codes=failure_codes,
                distance_score=distance_score,
                replay_tag=replay_tag,
            )

        # [4] 보상 계산 (살아남은 전략들만 - Scoring)
        # ================================================
        
        # A. Alpha Return (Tanh Saturation) - [V12.3] Smoother gradient
        return_scale = self.return_scale 
        r_return = np.tanh(alpha / max(1.0, return_scale)) * 3.0
        
        # B. 손익비(R:R) 보상
        if reward_risk >= self.excellent_rr:
            r_rr = 1.0
        elif reward_risk >= self.target_rr:
            r_rr = 0.5 + (reward_risk - self.target_rr) / (self.excellent_rr - self.target_rr) * 0.5
        elif reward_risk >= 1.0:
            r_rr = (reward_risk - 1.0) / (self.target_rr - 1.0) * 0.5
        else:
            r_rr = -self._clip((1.0 - reward_risk) * 2.0)
        r_rr = self._clip(r_rr)
        
        # C. 상위 트레이드 기여도 보상 (SSOT: top1_share 사용)
        # 0.2 (20%) 가 넘지 않는 것을 목표로 함 (집중도 낮추기)
        top1 = metrics.get("top1_share", 0.0)
        
        # top1_share가 높을수록 패널티, 낮을수록 보상
        if top1 <= 0.2:
            r_top_trades = 1.0
        elif top1 <= 0.4:
            r_top_trades = 0.5
        else:
            r_top_trades = 0.0
        r_top_trades = self._clip(r_top_trades)
        
        # D. Regime & Activity (Separated)
        
        # 1. Activity (Trading Frequency) - Quantity
        # Pure volume of trades relative to target
        if n_trades >= self.target_trades:
            r_trades = 1.0
        else:
            # [V12.3] Stage 1 Oxygen Supply: Stronger reward for initial trades
            if current_stage == 1:
                 # Concave function to reward first few trades heavily
                 # x / (x + k) form? Or simple sqrt?
                 # Let's use sqrt scaling for Stage 1
                 r_trades = np.sqrt(n_trades / max(1.0, float(self.target_trades)))
            else:
                 r_trades = n_trades / max(1.0, float(self.target_trades))
        r_trades = self._clip(r_trades, 0.0, 1.0)

        # 2. Regime Alignment (Trend Following) - Quality
        # Percentage of trades executed in alignment with market regime
        r_regime = 0.0
        regime_aligned_trades = metrics.get("regime_aligned_trades", 0)
        
        if n_trades > 0:
            regime_ratio = regime_aligned_trades / n_trades
            # [Design] Linear scaling: 100% aligned = 1.0, 0% = 0.0
            r_regime = regime_ratio
        
        r_regime = self._clip(r_regime, 0.0, 1.0)
        
        # E. 리스크 패널티
        r_mdd = 0.0
        if mdd_pct > self.mdd_threshold:
            r_mdd = -self._clip((mdd_pct - self.mdd_threshold) / self.mdd_threshold)

        r_winrate_penalty = 0.0
        if win_rate > self.winrate_penalty_start:
            excess = (win_rate - self.winrate_penalty_start) / (1.0 - self.winrate_penalty_start)
            r_winrate_penalty = -self._clip(excess) * 1.0
            
        # [V14] F. Complexity Penalty
        # Encourage parsimony: more features/depth = higher penalty
        # [V17] Use structural complexity from LogicTree if available
        comp_score = metrics.get("complexity_score", 0.0)
        if comp_score > 0:
            # Scale: 10 logical terms/nodes ~ 0.3 penalty
            r_complexity = -self._clip(comp_score * 0.03, 0, 0.4)
        else:
            genome = metrics.get("genome", {})
            n_features = len([k for k in genome.keys() if not k.startswith("__")])
            r_complexity = -self._clip(n_features * 0.05, 0, 0.3)

        # [V12.3] G. Anti-Luck Penalty (Soft Mode)
        # Hard mode is handled in Gates. Soft mode reduces reward here.
        r_antiluck = 0.0
        if getattr(config, "ANTILUCK_MODE", "soft") == "soft":
            top1 = metrics.get("top1_share", 0.0)
            if top1 > config.ANTILUCK_TOP1_SHARE_MAX: # e.g. 0.6
                 excess = top1 - config.ANTILUCK_TOP1_SHARE_MAX
                 r_antiluck = -self._clip(excess * 5.0) # Strong penalty for luck

        # [vAlpha+ / Genome v2] Phase 4: Information Contribution Focus
        # 1. Repeatability (Based on Alpha Variance across windows)
        alphas = [w.avg_alpha for w in metrics.get("window_results", [])]
        if len(alphas) > 1:
            var_alpha = np.var(alphas)
            avg_alpha = np.mean(alphas)
            
            # Information Efficiency: Reward high alpha with LOW variance
            # If variance is near zero, boost reward. If high, penalize.
            info_efficiency = avg_alpha / (np.sqrt(var_alpha) + 1e-6)
            r_repeat = np.tanh(info_efficiency / 2.0)
            
            # 2. Uncertainty Reduction (Contribution to stability)
            # If this is a complex strategy, it MUST have lower variance than simple ones
            complexity = metrics.get("complexity", 1.0)
            stability_premium = (1.0 - (var_alpha / 10.0)) * (complexity / 2.0)
            r_stability = self._clip(stability_premium, -1.0, 1.0)
        else:
            r_repeat = 0.5 # Neutral for single window
            r_stability = 0.0
            
        # 3. Cost Survival Factor (Economic Viability)
        cost_bps = config.DEFAULT_COST_BPS
        est_cost = (n_trades * cost_bps * 0.01)
        if est_cost > 0:
            survival_ratio = total_return_pct / est_cost
            r_cost_survival = np.tanh(survival_ratio / 3.0)
        else:
            r_cost_survival = 0.0

        # [Genome v2] Stability log
        # Assuming `batch_idx` and `logger` are available in this scope,
        # or need to be passed/imported. For now, assuming they are.
        # If not, this line will cause a NameError.
        # For a self-contained class method, these would typically be class attributes
        # or passed as arguments.
        # For the purpose of this edit, I'll include it as provided.
        # if batch_idx % 10 == 0:
        #     logger.info(f"    [Genome v2] Info Gain: {r_repeat:.2f} | Stability: {r_stability:.2f} | Survival: {r_cost_survival:.2f}")

        # [V16] AutoTuner Dynamic Weights
        from src.l3_meta.auto_tuner import get_auto_tuner
        tuner = get_auto_tuner()
        levers = tuner.get_levers()
        
        w_return = levers.get("reward_w_return", self.w_return)
        w_rr = levers.get("reward_w_rr", self.w_rr)
        w_top_trades = levers.get("reward_w_top_trades", self.w_top_trades)
        w_trades = levers.get("reward_w_trades", self.w_trades)
        w_mdd = levers.get("reward_w_mdd", self.w_mdd)

        # [vAlpha+] EAGL Weights
        w_repeat = getattr(config, "REWARD_REPEATABILITY_W", 0.2)
        w_cost_surv = getattr(config, "REWARD_COST_SURVIVAL_W", 0.3)
        w_stability = getattr(config, "REWARD_STABILITY_W", 0.2)

        # [합산]
        total = (
            w_return * r_return
            + w_rr * r_rr
            + w_top_trades * r_top_trades
            + w_trades * r_trades
            + w_mdd * r_mdd
            + r_winrate_penalty
            + r_complexity
            + r_antiluck
            + w_repeat * r_repeat
            + w_cost_surv * r_cost_survival
            + w_stability * r_stability
        )

        if failures:
            soft_scale = float(getattr(config, "REJECT_SOFT_PENALTY_SCALE", 0.5))
            soft_failures = [f for f in failures if not f.is_hard]
            if soft_failures:
                soft_reason = float(sum(reason_penalties.get(f.code, 0.0) for f in soft_failures))
                soft_dist = float(sum(distance_weights.get(f.code, 0.0) * f.distance for f in soft_failures))
                total += (soft_reason + soft_dist) * soft_scale

        return RewardBreakdown(
            total=float(total),
            return_component=float(r_return),
            rr_component=float(r_rr),
            top_trades_component=float(r_top_trades),
            regime_trade_component=float(r_regime),
            trades_component=float(r_trades),
            mdd_penalty=float(r_mdd),
            winrate_penalty=float(r_winrate_penalty),
            is_rejected=is_rejected,
            rejection_reason=rejection_reason,
            raw_metrics=metrics,
            alpha=float(alpha),
            failure_codes=failure_codes,
            distance_score=distance_score,
            replay_tag="PASS",
            repeatability_component=float(r_repeat),
            cost_survival_component=float(r_cost_survival),
            stability_component=float(r_stability)
        )
    
    
    def compute_from_cpcv(
        self,
        cpcv_metrics: Dict[str, float],
        pbo: float = 0.0,
    ) -> float:
        """
        CPCV 지표에서 보상을 계산합니다.
        
        cpcv_metrics에는 Backtest 결과가 포함되어 있어야 합니다.
        """
        # Backtest 결과 직접 사용
        metrics = {
            "total_return_pct": cpcv_metrics.get("total_return_pct", 0.0),
            "mdd_pct": cpcv_metrics.get("mdd_pct", 0.0),
            "n_trades": cpcv_metrics.get("n_trades", 0),
            "win_rate": cpcv_metrics.get("win_rate", 0.5),
            "reward_risk": cpcv_metrics.get("reward_risk", 1.0),
            "trade_returns": cpcv_metrics.get("trade_returns", None),
            "regime_aligned_trades": cpcv_metrics.get("regime_aligned_trades", None),
        }
        
        # PBO 페널티 (높을수록 과적합)
        pbo_penalty = 0.0
        if pbo > 0.3:
            pbo_penalty = (pbo - 0.3) * 0.5
        
        reward = self.compute(metrics) - pbo_penalty
        
        return float(reward)
    
    def _clip(self, value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """값을 범위 내로 클리핑합니다."""
        return float(np.clip(value, min_val, max_val))
    
    @property
    def weights(self) -> Dict[str, float]:
        """현재 가중치를 반환합니다."""
        return {
            "return": self.w_return,
            "rr": self.w_rr,
            "top_trades": self.w_top_trades,
            "regime_trade": self.w_regime_trade,
            "trades": self.w_trades,
            "mdd": self.w_mdd,
        }


# 전역 인스턴스
_reward_shaper: Optional[RewardShaper] = None


def get_reward_shaper() -> RewardShaper:
    """RewardShaper 싱글톤을 반환합니다."""
    global _reward_shaper
    if _reward_shaper is None:
        _reward_shaper = RewardShaper()
    return _reward_shaper


def reset_reward_shaper() -> None:
    """테스트용 리셋."""
    global _reward_shaper
    _reward_shaper = None


def compute_reward(metrics: Dict[str, float]) -> float:
    """평가 지표에서 보상을 계산합니다 (편의 함수)."""
    return get_reward_shaper().compute(metrics)


def compute_reward_from_cpcv(cpcv_metrics: Dict[str, float], pbo: float = 0.0) -> float:
    """CPCV 지표에서 보상을 계산합니다 (편의 함수)."""
    return get_reward_shaper().compute_from_cpcv(cpcv_metrics, pbo)
