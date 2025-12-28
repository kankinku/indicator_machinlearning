from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.contracts import PolicySpec
from src.config import config
from src.shared.logger import get_logger

logger = get_logger("l1.diversity")

def calculate_genome_similarity(p1: PolicySpec, p2: PolicySpec) -> float:
    """지표 집합의 Jaccard 유사도를 계산합니다."""
    g1 = set(p1.feature_genome.keys())
    g2 = set(p2.feature_genome.keys())
    
    if not g1 and not g2:
        return 1.0
    
    intersection = g1.intersection(g2)
    union = g1.union(g2)
    
    jaccard = len(intersection) / len(union)
    return jaccard

def calculate_param_similarity(p1: PolicySpec, p2: PolicySpec) -> float:
    """공통 지표에 대한 파라미터 유사도를 계산합니다."""
    g1 = p1.feature_genome
    g2 = p2.feature_genome
    
    intersection = set(g1.keys()).intersection(set(g2.keys()))
    if not intersection:
        return 0.0
    
    total_dist = 0.0
    count = 0
    
    for feat_id in intersection:
        params1 = g1[feat_id]
        params2 = g2[feat_id]
        
        # 공통 파라미터만 비교
        common_params = set(params1.keys()).intersection(set(params2.keys()))
        for p_name in common_params:
            v1 = params1[p_name]
            v2 = params2[p_name]
            
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # 간단한 정규화 거리 (차이의 절대값 / 큰 값)
                denom = max(abs(v1), abs(v2), 1e-6)
                dist = abs(v1 - v2) / denom
                total_dist += dist
                count += 1
            elif v1 != v2:
                # 범주형 파라미터가 다르면 최대 거리
                total_dist += 1.0
                count += 1
                
    if count == 0:
        return 0.0
        
    avg_dist = total_dist / count
    # 유사도 = 1 - 평균 거리
    return max(0.0, 1.0 - avg_dist)

def compute_robustness_score(candidate: Any, stage_cfg: object) -> float:
    window_results = getattr(candidate, "window_results", []) or []
    if not window_results:
        return 0.0

    alpha_floor = float(getattr(stage_cfg, "alpha_floor", -10.0))
    consistency = np.mean([1.0 if w.avg_alpha >= alpha_floor else 0.0 for w in window_results])
    violation_rate = np.mean([w.violation_rate for w in window_results])
    return float(consistency - violation_rate)


def compute_selection_score(candidate: Any, gate_diag: object, stage_cfg: object) -> float:
    reward_breakdown = getattr(candidate, "reward_breakdown", None)
    reward_total = None
    if isinstance(reward_breakdown, dict):
        reward_total = reward_breakdown.get("total")
    performance = float(reward_total if reward_total is not None else getattr(candidate, "score", 0.0))
    progress = float(getattr(gate_diag, "soft_gate_score", 0.0))
    robustness = compute_robustness_score(candidate, stage_cfg)

    w_perf = float(getattr(config, "SELECTION_W_PERFORMANCE", 1.0))
    w_prog = float(getattr(config, "SELECTION_W_PROGRESS", 1.0))
    w_rob = float(getattr(config, "SELECTION_W_ROBUSTNESS", 1.0))

    return (w_perf * performance) + (w_prog * progress) + (w_rob * robustness)


def select_diverse_top_k(
    candidates: List[Any], # List[ModuleResult] or List[ExperimentResult]
    k: int,
    jaccard_th: float = 0.7,
    param_th: float = 0.1, # Not used as similarity but distance if we want
    gate_diag_map: Optional[Dict[str, object]] = None,
    seed: Optional[List[Any]] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    유사도 제약을 고려하여 상위 K개를 선택합니다.
    Returns: (selected_list, stats_dict)
    """
    if not candidates:
        return [], {"collision_count": 0, "avg_jaccard": 0.0}
        
    from src.l1_judge.evaluator import compute_gate_diagnostics

    stage_id = getattr(config, "CURRICULUM_CURRENT_STAGE", 1)
    stage_cfg = config.CURRICULUM_STAGES.get(stage_id, {})
    diversity_weight = float(getattr(config, "SELECTION_W_DIVERSITY", 0.0))

    selected = list(seed or [])
    if len(selected) >= k:
        return selected[:k], {"collision_count": 0, "avg_jaccard": 0.0, "collision_rate": 0.0}

    base_scores: Dict[int, float] = {}
    for cand in candidates:
        metrics = getattr(getattr(cand, "best_sample", None), "metrics", None)
        if metrics is None:
            base_scores[id(cand)] = float(getattr(cand, "score", 0.0))
            continue

        gate_diag = None
        if gate_diag_map:
            policy = getattr(cand, "policy_spec", getattr(cand, "policy", None))
            if policy is not None:
                gate_diag = gate_diag_map.get(policy.spec_id)
        if gate_diag is None:
            gate_diag = compute_gate_diagnostics(metrics)

        base_scores[id(cand)] = compute_selection_score(cand, gate_diag, stage_cfg)

    remaining = [c for c in candidates if c not in selected]
    collision_count = 0
    all_jaccards = []
    
    while remaining and len(selected) < k:
        best_idx = None
        best_score = -float("inf")

        for idx, cand in enumerate(remaining):
            p_cand = getattr(cand, "policy_spec", getattr(cand, "policy", None))
            if p_cand is None:
                continue

            is_too_similar = False
            max_jaccard = 0.0

            for sel in selected:
                p_sel = getattr(sel, "policy_spec", getattr(sel, "policy", None))
                if p_sel is None:
                    continue

                jaccard = calculate_genome_similarity(p_cand, p_sel)
                all_jaccards.append(jaccard)
                max_jaccard = max(max_jaccard, jaccard)

                if jaccard > jaccard_th:
                    param_sim = calculate_param_similarity(p_cand, p_sel)
                    if param_sim > (1.0 - param_th):
                        is_too_similar = True
                        collision_count += 1
                        break

            if is_too_similar:
                continue

            diversity_bonus = diversity_weight * (1.0 - max_jaccard) if selected else 0.0
            effective_score = base_scores.get(id(cand), 0.0) + diversity_bonus
            if effective_score > best_score:
                best_score = effective_score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(remaining.pop(best_idx))
            
    stats = {
        "collision_count": collision_count,
        "avg_jaccard": np.mean(all_jaccards) if all_jaccards else 0.0,
        "collision_rate": collision_count / len(candidates) if candidates else 0
    }
    
    return selected, stats
