from __future__ import annotations
import time
import copy
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import numpy as np
from src.config import config
from src.shared.logger import get_logger
from src.shared.event_bus import record_event

logger = get_logger("l3.auto_tuner")

@dataclass
class BatchState:
    batch_id: int
    timestamp: float
    best_score: float
    mean_topk_score: float
    pass_rate_s1: float
    pass_rate_s2: float
    trade_count_mean_topk: float
    zero_trade_ratio_topk: float
    diversity_mean_topk: float
    duplicate_ratio_topk: float
    exception_count: int
    regime: str
    search_profile_mix: Dict[str, float] = field(default_factory=dict)

@dataclass
class InterventionPlan:
    intervention_id: str
    cause: str
    levers: Dict[str, Any]
    start_batch: int
    duration: int
    expected_metrics: List[str]
    initial_values: Dict[str, Any]
    status: str = "ACTIVE" # ACTIVE, SUCCESS, FAILED, ROLLBACK

class AutoTuner:
    """
    [V16] Advanced Self-Correcting Auto-Tuner
    Classifies "RIGID" states into specific causes and applies targeted interventions.
    """
    def __init__(self):
        self.history: List[BatchState] = []
        self.interventions: List[InterventionPlan] = []
        self.ema_best_score = None
        self.ema_topk_score = None
        
        # Levers tracked by tuner
        self.current_levers = {
            "mutation_strength": 0.3,
            "diversity_pressure": 0.8,
            "s1_threshold_adjust": 0.0,
            "reward_w_trades": config.REWARD_W_TRADES,
            "reward_w_mdd": config.REWARD_W_MDD,
            "reward_w_return": config.REWARD_W_RETURN,
            "search_profile_mix": None, # If None, use agent default
            "indicator_swap_prob": config.AUTOTUNE_INDICATOR_SWAP_PROB
        }

    def process_diagnostics(self, diag: Dict[str, Any], extra_info: Dict[str, Any]):
        """
        Unified entry point from infinite_loop.
        Maps OperationalQACollector report to BatchState.
        """
        try:
            state = BatchState(
                batch_id=extra_info.get("batch_id", 0),
                timestamp=diag.get("timestamp", time.time()),
                best_score=diag.get("best_score", config.EVAL_SCORE_MIN),
                mean_topk_score=np.mean(diag.get("performance_stats", {}).get("scores", [config.EVAL_SCORE_MIN])),
                pass_rate_s1=extra_info.get("pass_rate_s1", 0.0),
                pass_rate_s2=diag.get("pass_rate", 0.0),
                trade_count_mean_topk=diag.get("mean_tpy", 0.0),
                zero_trade_ratio_topk=diag.get("zero_trade_ratio", 0.0),
                diversity_mean_topk=1.0 - diag.get("similarity", {}).get("avg_jaccard", 0.0),
                duplicate_ratio_topk=diag.get("similarity", {}).get("collision_rate", 0.0),
                exception_count=extra_info.get("exception_count", 0),
                regime=diag.get("regime", "UNKNOWN"),
                search_profile_mix=extra_info.get("search_profile_mix", {})
            )
            self.record_batch(state)
        except Exception as e:
            logger.error(f"[AutoTuner] Failed to process diagnostics: {e}")

    def record_batch(self, state: BatchState):
        self.history.append(state)
        if len(self.history) > config.AUTOTUNE_HISTORY_SIZE:
            self.history.pop(0)
            
        # Update EMA
        if self.ema_best_score is None:
            self.ema_best_score = state.best_score
            self.ema_topk_score = state.mean_topk_score
        else:
            alpha = 0.2
            self.ema_best_score = (1-alpha) * self.ema_best_score + alpha * state.best_score
            self.ema_topk_score = (1-alpha) * self.ema_topk_score + alpha * state.mean_topk_score

        self._check_interventions(state.batch_id)
        self._analyze_and_act(state)
        
        # Log status
        active = [p for p in self.interventions if p.status == "ACTIVE"]
        if active:
            logger.info(f"[AutoTuner] Active Intervention: {active[0].intervention_id} ({active[0].cause})")

    def _check_interventions(self, current_batch: int):
        """Verifies if active interventions met their goals or need rollback."""
        for plan in self.interventions:
            if plan.status == "ACTIVE" and current_batch >= plan.start_batch + plan.duration:
                # Evaluation time
                success = self._evaluate_intervention(plan)
                if success:
                    plan.status = "SUCCESS"
                    record_event("AUTOTUNER_INTERVENTION_SUCCESS", payload={"plan_id": plan.intervention_id, "cause": plan.cause})
                    logger.info(f"[AutoTuner] ✅ Intervention {plan.intervention_id} SUCCESS. Maintaining levers.")
                else:
                    plan.status = "FAILED"
                    record_event("AUTOTUNER_INTERVENTION_FAILED", payload={"plan_id": plan.intervention_id, "cause": plan.cause})
                    logger.warning(f"[AutoTuner] ❌ Intervention {plan.intervention_id} FAILED. Rolling back.")
                    self._rollback(plan)
                    record_event("AUTOTUNER_INTERVENTION_ROLLBACK", payload={"plan_id": plan.intervention_id})

    def _analyze_and_act(self, current: BatchState):
        """Root cause analysis and intervention strategy selection."""
        # Need at least a few batches to see trends
        if len(self.history) < 3:
            return

        cause = self._classify_cause(current)
        if cause == "HEALTHY":
            return
        
        record_event("AUTOTUNER_RIGID_DETECTED", payload={"cause": cause})

        # Start new intervention if not already tracking an active one
        if any(p.status == "ACTIVE" for p in self.interventions):
            return

        self._apply_intervention(cause, current.batch_id)
        record_event("AUTOTUNER_INTERVENTION_APPLIED", payload={"cause": cause})

    def _classify_cause(self, current: BatchState) -> str:
        """Heuristic-based Cause Classification (A-F)."""
        from src.testing.context import is_test_mode, get_test_context
        if is_test_mode():
            ctx = get_test_context()
            if ctx and ctx.env and ctx.env.force_rigid:
                return "SPACE_TOO_NARROW" # Default forced rigid cause
        
        # 0. Infrastructure check (F)
        if current.exception_count > 5:
            return "INFRA_NOISE"

        # 1. Rigidity/Stagnation check
        # Best score improvement over N batches
        best_scores = [s.best_score for s in self.history]
        recent_best = np.max(best_scores[-3:])
        past_best = np.max(best_scores[:-3]) if len(best_scores) > 3 else best_scores[0]
        improvement = recent_best - past_best
        
        is_rigid = improvement < config.AUTOTUNE_RIGID_THRESHOLD
        
        if is_rigid:
            # 원인 A: 필터/게이트 과보수 (Pass Rate Too Low)
            if current.pass_rate_s1 < 0.10 or current.pass_rate_s2 < 0.05:
                return "PASS_RATE_TOO_LOW"
            
            # 원인 B: 유전적 다양성 붕괴
            if current.duplicate_ratio_topk > 0.4 or current.diversity_mean_topk < config.AUTOTUNE_DIVERSITY_TARGET:
                return "DIVERSITY_COLLAPSE"
                
            # 원인 D: 거래 없음
            if current.zero_trade_ratio_topk > 0.4 or current.trade_count_mean_topk < 5:
                return "NO_TRADE"
                
            # 원인 E: 레짐 불일치 (최근 레짐 변화가 있었는지 체크 - 단순화)
            if len(self.history) >= 2 and self.history[-1].regime != self.history[-2].regime:
                return "REGIME_MISMATCH"

            # 원인 C: 탐색 공간 좁음
            return "SPACE_TOO_NARROW"

        # 붕괴 체크 (COLLAPSE)
        if current.pass_rate_s2 < 0.01 and current.best_score < config.EVAL_SCORE_MIN * 0.5:
            return "COLLAPSE"

        return "HEALTHY"

    def _apply_intervention(self, cause: str, batch_id: int):
        plan_id = f"INT-{batch_id}-{cause[:4]}"
        levers = {}
        expected = []
        initial = {}

        if cause == "PASS_RATE_TOO_LOW":
            # 조치: 임계치 완화, epsilon 소폭 증가
            levers = {
                "s1_threshold_adjust": self.current_levers["s1_threshold_adjust"] - 0.05,
                "reward_w_mdd": self.current_levers["reward_w_mdd"] * 0.8 # Relax MDD penalty
            }
            expected = ["pass_rate_s1", "pass_rate_s2"]
            
        elif cause == "DIVERSITY_COLLAPSE":
            # 조치: 다양성 압력 및 돌연변이 강화
            levers = {
                "diversity_pressure": min(1.0, self.current_levers["diversity_pressure"] + 0.1),
                "mutation_strength": min(1.0, self.current_levers["mutation_strength"] + 0.2),
                "indicator_swap_prob": 0.5
            }
            expected = ["diversity_mean_topk", "duplicate_ratio_topk"]
            
        elif cause == "NO_TRADE":
            # 조치: 거래수 페널티 강화, 적극적 프로파일 유도
            levers = {
                "reward_w_trades": self.current_levers["reward_w_trades"] * 2.0,
                "search_profile_mix": {"SCALPING_FAST": 0.6, "TREND": 0.4} # Hardcoded for now
            }
            expected = ["zero_trade_ratio_topk", "trade_count_mean_topk"]
            
        elif cause == "SPACE_TOO_NARROW":
            # 조치: indicator pool 확장 유도 (mutation 강도 및 유형 조절)
            levers = {
                "mutation_strength": 0.7,
                "indicator_swap_prob": 0.8
            }
            expected = ["best_score", "diversity_mean_topk"]
            
        elif cause == "REGIME_MISMATCH":
            # 조치: 프로파일 믹스 분산 (Exploration 강화)
            levers = {
                "search_profile_mix": {"SCALPING_FAST": 0.25, "MEANREV": 0.25, "TREND": 0.25, "DEFENSIVE": 0.25}
            }
            expected = ["best_score"]

        elif cause == "INFRA_NOISE":
            # 조치: 안전장치 (여기서는 로깅 및 임계치 소폭 조정만)
            levers = {"s1_threshold_adjust": 0.0} # Reset
            expected = ["exception_count"]
            
        elif cause == "COLLAPSE":
            # 조치: 긴급 대피 (Epsilon Reheat 유도 등은 Agent가 하겠지만 여기서도 거듦)
            levers = {
                "s1_threshold_adjust": 0.0,
                "mutation_strength": 0.8,
                "reward_w_mdd": config.REWARD_W_MDD * 1.5 # Protect
            }
            expected = ["pass_rate_s2", "best_score"]
        
        if not levers:
            return

        # Snapshot initials and apply
        for k, v in levers.items():
            initial[k] = self.current_levers.get(k)
            self.current_levers[k] = v

        plan = InterventionPlan(
            intervention_id=plan_id,
            cause=cause,
            levers=levers,
            start_batch=batch_id,
            duration=config.AUTOTUNE_INTERVENTION_M_BATCHES,
            expected_metrics=expected,
            initial_values=initial
        )
        self.interventions.append(plan)
        logger.warning(f"[AutoTuner] 🚀 New Intervention Applied: {plan_id}\nCause: {cause}\nLevers: {levers}")

    def _evaluate_intervention(self, plan: InterventionPlan) -> bool:
        """Compares current state with state before intervention."""
        if len(self.history) < 2:
            return True
            
        current = self.history[-1]
        past = None
        for s in reversed(self.history[:-1]):
            if s.batch_id <= plan.start_batch:
                past = s
                break
        
        if not past:
            return True # Not enough data to fail

        # Check if any expected metric improved
        improvements = 0
        for metric in plan.expected_metrics:
            curr_val = getattr(current, metric, 0)
            past_val = getattr(past, metric, 0)
            
            # Directional success check
            if metric in ["pass_rate_s1", "pass_rate_s2", "best_score", "diversity_mean_topk", "trade_count_mean_topk"]:
                if curr_val > past_val: improvements += 1
            if metric in ["duplicate_ratio_topk", "zero_trade_ratio_topk", "exception_count"]:
                if curr_val < past_val: improvements += 1
                
        # Also check if best score overall improved (ultimate goal)
        if current.best_score > past.best_score:
            improvements += 1
            
        return improvements > 0

    def _rollback(self, plan: InterventionPlan):
        for k, v in plan.initial_values.items():
            self.current_levers[k] = v
        logger.info(f"[AutoTuner] Rollback successful for {plan.intervention_id}")

    def get_levers(self) -> Dict[str, Any]:
        return self.current_levers

_tuner_instance = None

def get_auto_tuner() -> AutoTuner:
    global _tuner_instance
    if _tuner_instance is None:
        _tuner_instance = AutoTuner()
    return _tuner_instance
