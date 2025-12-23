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
from src.shared.logger import get_logger

logger = get_logger("l3.curriculum")


@dataclass
class StageRequirement:
    """Stage별 요구 사항."""
    stage_id: int
    min_trades: int
    max_mdd_pct: float
    min_winrate: float
    max_winrate: float
    min_alpha: float
    min_return_pct: float
    description: str = ""


@dataclass
class CurriculumState:
    """현재 커리큘럼 상태."""
    current_stage: int = 1
    stage_passes: int = 0  # 현재 Stage 통과 횟수
    stage_attempts: int = 0  # 현재 Stage 시도 횟수
    total_experiments: int = 0
    stage_history: List[Dict] = field(default_factory=list)


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
        self.state = self._load_state()
        self.stages = self._build_stages()
        
    def _build_stages(self) -> Dict[int, StageRequirement]:
        """Config에서 Stage 요구사항을 빌드합니다."""
        stages = {}
        for stage_id, stage_config in config.CURRICULUM_STAGES.items():
            stages[stage_id] = StageRequirement(
                stage_id=stage_id,
                min_trades=stage_config.get("min_trades", 10),
                max_mdd_pct=stage_config.get("max_mdd_pct", 60.0),
                min_winrate=stage_config.get("min_winrate", 0.25),
                max_winrate=stage_config.get("max_winrate", 0.90),
                min_alpha=stage_config.get("min_alpha", 0.0),
                min_return_pct=stage_config.get("target_return_pct", 50.0),
                description=stage_config.get("description", f"Stage {stage_id}"),
            )
        return stages
    
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
    def current_requirement(self) -> StageRequirement:
        """현재 Stage의 요구사항을 반환합니다."""
        return self.stages.get(
            self.state.current_stage,
            self.stages.get(1, StageRequirement(stage_id=1, min_return_pct=30.0, min_trades=30, max_mdd_pct=50.0, min_winrate=0.4, max_winrate=0.8, min_alpha=0.0))
        )
    
    def get_previous_requirement(self) -> Optional[StageRequirement]:
        """이전 Stage의 요구사항을 반환합니다."""
        prev_stage = self.state.current_stage - 1
        return self.stages.get(prev_stage)
    
    def evaluate_against_current(
        self,
        total_return_pct: float,
        n_trades: int,
        mdd_pct: float,
        win_rate: float,
        alpha: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        [V11.2] 현재 Stage 기준으로 전략을 평가합니다. (Hard Gate + Alpha)
        """
        req = self.current_requirement
        
        reasons = []
        passed = True
        
        # 1. 최소 거래
        if n_trades < req.min_trades:
            passed = False
            reasons.append(f"LOW_TRADES ({n_trades} < {req.min_trades})")
        
        # 2. 최대 MDD
        if abs(mdd_pct) > req.max_mdd_pct:
            passed = False
            reasons.append(f"MDD ({abs(mdd_pct):.1f}% > {req.max_mdd_pct}%)")
        
        # 3. 승률 범위
        if win_rate < req.min_winrate:
            passed = False
            reasons.append(f"WINRATE_LOW ({win_rate:.1%} < {req.min_winrate:.1%})")
        elif win_rate > req.max_winrate:
            passed = False
            reasons.append(f"WINRATE_HIGH ({win_rate:.1%} > {req.max_winrate:.1%})")
            
        # 4. 최소 Alpha (벤치마크 초과 수익)
        if alpha < req.min_alpha:
            passed = False
            reasons.append(f"ALPHA_NEGATIVE ({alpha:.4f} < {req.min_alpha:.4f})")
            
        # 5. 목표 수익률 달성 (승급 기여용 보조 체크)
        if total_return_pct < req.min_return_pct:
            passed = False
            reasons.append(f"TARGET_RETURN ({total_return_pct:.1f}% < {req.min_return_pct}%)")
        
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
        
        # 승급 조건: N개 이상 통과
        threshold = config.CURRICULUM_STAGE_UP_THRESHOLD
        max_stage = max(self.stages.keys())
        
        if self.state.stage_passes >= threshold and self.state.current_stage < max_stage:
            # 승급!
            old_stage = self.state.current_stage
            self.state.current_stage += 1
            self.state.stage_passes = 0
            self.state.stage_attempts = 0
            self.state.stage_history.append({
                "stage": old_stage,
                "experiments": self.state.total_experiments,
                "action": "promoted",
            })
            status_change["promoted"] = True
            logger.info(f">>> [Curriculum] PROMOTED to Stage {self.state.current_stage}!")
        
        status_change["stage_after"] = self.state.current_stage
        status_change["pass_rate"] = (
            self.state.stage_passes / self.state.stage_attempts 
            if self.state.stage_attempts > 0 else 0.0
        )
        
        self._save_state()
        return status_change
    
    def get_dynamic_min_trades(self) -> int:
        """현재 Stage에 따른 동적 최소 거래 수를 반환합니다."""
        return self.current_requirement.min_trades
    
    def get_dynamic_min_return(self) -> float:
        """현재 Stage에 따른 동적 최소 수익률을 반환합니다."""
        return self.current_requirement.min_return_pct
    
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
            "min_return_pct": req.min_return_pct,
            "min_trades": req.min_trades,
            "description": req.description,
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
