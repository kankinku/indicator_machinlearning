"""
Curriculum Controller - 단계별 학습 관리자

설계도 기준:
- 500%는 최종 목표이지 출발점이 아니다
- Stage 통과율/세대 수 기준 자동 승급
- 이전 Stage 기준 미달 시 탈락

Stages:
1: +30~50%  | 기본 활동성 확보
2: +70~100% | 수익성 개선
3: +150%    | 안정적 수익
4: +300~500%| 고성능 전략
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import os

from src.config import config
from src.shared.stage_schema import StageSpec
from src.shared.logger import get_logger

logger = get_logger("l3.curriculum")


# StageRequirement is replaced by StageSpec in shared/stage_schema.py


@dataclass
class CurriculumState:
    """현재 커리큘럼 상태."""
    current_stage: int = 1
    stage_passes: int = 0  # 현재 Stage 통과 횟수
    stage_attempts: int = 0  # 현재 Stage 시도 횟수
    total_experiments: int = 0
    stage_history: List[Dict] = field(default_factory=list)
    
    # [V18] Performance Rolling Stats
    rolling_alpha: float = -1.0
    rolling_win_rate: float = 0.0
    rolling_window: int = 20



class CurriculumController:
    """
    단계별 학습 목표를 관리합니다.
    
    원칙:
    1. 처음부터 500%를 요구하지 않음
    2. Stage별 점진적 목표 상향
    3. 통과율 기준 자동 승급
    4. 이전 Stage 기준 미달 시 탈락
    """
    
    def __init__(self, state_file: Optional[str] = None):
        self.state_file = state_file or str(config.LEDGER_DIR / "curriculum_state.json")
        self._audit_schema()
        self.state = self._load_state()
        self.stages = config.CURRICULUM_STAGES # Direct use of StageSpec objects
        
    def _audit_schema(self) -> None:
        """[V15] Boot-time Schema Audit."""
        logger.info("--- [Curriculum] SCHEMA AUDIT ---")
        for sid, spec in config.CURRICULUM_STAGES.items():
            logger.info(
                f"Stage {sid} ({spec.name}): "
                f"Ret={spec.target_return_pct}% | "
                f"Alpha={spec.alpha_floor} | "
                f"Trades={spec.min_trades_per_year} | "
                f"MDD={spec.max_mdd_pct}% | "
                f"WF={spec.wf_splits} ({spec.wf_gate_mode})"
            )
        logger.info("---------------------------------")
    
    def _load_state(self) -> CurriculumState:
        """저장된 상태를 로드합니다."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                return CurriculumState(
                    current_stage=data.get("current_stage", 1),
                    stage_passes=data.get("stage_passes", 0),
                    stage_attempts=data.get("stage_attempts", 0),
                    total_experiments=data.get("total_experiments", 0),
                    stage_history=data.get("stage_history", []),
                    rolling_alpha=data.get("rolling_alpha", -1.0),
                    rolling_win_rate=data.get("rolling_win_rate", 0.0),
                )
            except Exception as e:
                logger.warning(f"Failed to load curriculum state: {e}")
        return CurriculumState()
    
    def _save_state(self) -> None:
        """현재 상태를 저장합니다."""
        try:
            data = {
                "current_stage": self.state.current_stage,
                "stage_passes": self.state.stage_passes,
                "stage_attempts": self.state.stage_attempts,
                "total_experiments": self.state.total_experiments,
                "stage_history": self.state.stage_history,
                "rolling_alpha": self.state.rolling_alpha,
                "rolling_win_rate": self.state.rolling_win_rate,
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save curriculum state: {e}")
    
    @property
    def current_stage(self) -> int:
        """현재 Stage를 반환합니다."""
        return self.state.current_stage
    
    @property
    def current_requirement(self) -> StageSpec:
        """현재 Stage의 요구사항을 반환합니다."""
        return self.stages.get(
            self.state.current_stage,
            list(self.stages.values())[0] # Fallback to first stage
        )
    
    def get_previous_requirement(self) -> Optional[StageSpec]:
        """이전 Stage의 요구사항을 반환합니다."""
        prev_stage = self.state.current_stage - 1
        return self.stages.get(prev_stage)
    
    def evaluate_against_current(
        self,
        total_return_pct: float,
        trades_per_year: float,
        mdd_pct: float,
        win_rate: float,
        alpha: float = 0.0,
        profit_factor: float = 1.0,
    ) -> Tuple[bool, str]:
        """
        [V15] 현재 Stage 기준으로 전략을 평가합니다. (StageSpec SSOT 사용)
        """
        req = self.current_requirement
        
        reasons = []
        passed = True
        
        # 1. 최소 거래 (연간 기준)
        if trades_per_year < req.min_trades_per_year:
            passed = False
            reasons.append(f"LOW_TRADES ({trades_per_year:.1f} < {req.min_trades_per_year:.1f})")
        
        # 2. 최대 MDD
        if abs(mdd_pct) > req.max_mdd_pct:
            passed = False
            reasons.append(f"MDD ({abs(mdd_pct):.1f}% > {req.max_mdd_pct}%)")
            
        # 3. 최소 Alpha (벤치마크 초과 수익)
        if alpha < req.alpha_floor:
            passed = False
            reasons.append(f"ALPHA_BELOW_FLOOR ({alpha:.2f} < {req.alpha_floor:.2f})")
            
        # 4. 목표 수익률 달성 (승급 기여용 보조 체크)
        if total_return_pct < req.target_return_pct:
            # Note: Stage 1에서는 target_return_pct 미달이어도 survival 조건(alpha_floor 등)을 만족하면 
            # evaluation score는 나올 수 있지만, curriculum PASS 여부에서는 걸러짐.
            passed = False
            reasons.append(f"TARGET_RETURN ({total_return_pct:.1f}% < {req.target_return_pct}%)")

        # 5. Profit Factor
        if profit_factor < req.min_profit_factor:
            passed = False
            reasons.append(f"LOW_PF ({profit_factor:.2f} < {req.min_profit_factor:.2f})")
        
        if passed:
            return True, f"PASS_STAGE_{req.stage_id}"
        else:
            return False, " | ".join(reasons)
    
    def record_result(self, passed: bool, metrics: Optional[Dict] = None) -> Dict:
        """
        실험 결과를 기록하고 Stage 승급 여부를 판단합니다.
        
        Returns:
            상태 변경 정보
        """
        self.state.total_experiments += 1
        self.state.stage_attempts += 1
        
        status_change = {
            "stage_before": self.state.current_stage,
            "promoted": False,
            "demoted": False,
        }
        
        if passed:
            self.state.stage_passes += 1
            
        # [vLearn+] Update Rolling Performance
        if metrics:
            alpha = float(metrics.get("excess_return", -1.0))
            wr = float(metrics.get("win_rate", 0.0))
            
            # Simple EMA Update (alpha=0.1)
            eta = 0.1
            if self.state.rolling_alpha <= -1.0: # Initial
                 self.state.rolling_alpha = alpha
                 self.state.rolling_win_rate = wr
            else:
                 self.state.rolling_alpha = (1 - eta) * self.state.rolling_alpha + eta * alpha
                 self.state.rolling_win_rate = (1 - eta) * self.state.rolling_win_rate + eta * wr

        # [vLearn+] Dynamic Promotion Logic
        threshold = config.CURRICULUM_STAGE_UP_THRESHOLD
        min_alpha = getattr(config, "CURRICULUM_ALPHA_PROMOTION_THRESHOLD", 0.01)
        max_stage = max(self.stages.keys())
        
        # Promotion Condition: 
        # (Pass count threshold reached) OR (Excellent Alpha Performance)
        can_promote = (
            (self.state.stage_passes >= threshold) or 
            (self.state.rolling_alpha > min_alpha and self.state.stage_attempts >= 30)
        )
        
        if can_promote and self.state.current_stage < max_stage:
            # 승급!
            old_stage = self.state.current_stage
            self.state.current_stage += 1
            self.state.stage_passes = 0
            self.state.stage_attempts = 0
            # Reset rolling on new stage to re-prove
            self.state.rolling_alpha = -1.0 
            
            self.state.stage_history.append({
                "stage": old_stage,
                "experiments": self.state.total_experiments,
                "action": "promoted",
                "final_rolling_alpha": self.state.rolling_alpha
            })
            status_change["promoted"] = True
            logger.info(f">>> [Curriculum] PROMOTED to Stage {self.state.current_stage} (Alpha: {self.state.rolling_alpha:.3f})!")
        
        status_change["stage_after"] = self.state.current_stage
        status_change["pass_rate"] = (
            self.state.stage_passes / self.state.stage_attempts 
            if self.state.stage_attempts > 0 else 0.0
        )
        
        self._save_state()
        return status_change
    
    def get_dynamic_min_trades(self) -> float:
        """현재 Stage에 따른 동적 최소 거래 수(연간)를 반환합니다."""
        return self.current_requirement.min_trades_per_year
    
    def get_dynamic_min_return(self) -> float:
        """현재 Stage에 따른 동적 목표 수익률을 반환합니다."""
        return self.current_requirement.target_return_pct
    
    def get_stage_info(self) -> Dict:
        """현재 Stage 정보를 반환합니다."""
        req = self.current_requirement
        return {
            "current_stage": self.state.current_stage,
            "stage_passes": self.state.stage_passes,
            "stage_attempts": self.state.stage_attempts,
            "pass_rate": (
                self.state.stage_passes / self.state.stage_attempts 
                if self.state.stage_attempts > 0 else 0.0
            ),
            "target_return_pct": req.target_return_pct,
            "min_trades_per_year": req.min_trades_per_year,
            "description": req.name,
            "threshold_to_next": config.CURRICULUM_STAGE_UP_THRESHOLD,
        }


# ============================================
# Singleton Instance
# ============================================
_curriculum_controller: Optional[CurriculumController] = None


def get_curriculum_controller() -> CurriculumController:
    """CurriculumController 싱글톤을 반환합니다."""
    global _curriculum_controller
    if _curriculum_controller is None:
        _curriculum_controller = CurriculumController()
    return _curriculum_controller


def reset_curriculum_controller() -> None:
    """테스트용 리셋."""
    global _curriculum_controller
    _curriculum_controller = None
