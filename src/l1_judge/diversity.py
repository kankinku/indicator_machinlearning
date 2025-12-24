from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Tuple
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

def select_diverse_top_k(
    candidates: List[Any], # List[ModuleResult] or List[ExperimentResult]
    k: int,
    jaccard_th: float = 0.7,
    param_th: float = 0.1, # Not used as similarity but distance if we want
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    유사도 제약을 고려하여 상위 K개를 선택합니다.
    Returns: (selected_list, stats_dict)
    """
    if not candidates:
        return [], {"collision_count": 0, "avg_jaccard": 0.0}
        
    # 점수순 정렬
    sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
    
    selected = []
    collision_count = 0
    all_jaccards = []
    
    for cand in sorted_candidates:
        if len(selected) >= k:
            break
            
        p_cand = getattr(cand, 'policy_spec', getattr(cand, 'policy', None))
        is_too_similar = False
        
        for sel in selected:
            p_sel = getattr(sel, 'policy_spec', getattr(sel, 'policy', None))
            
            if p_cand and p_sel:
                jaccard = calculate_genome_similarity(p_cand, p_sel)
                all_jaccards.append(jaccard)
                
                if jaccard > jaccard_th:
                    param_sim = calculate_param_similarity(p_cand, p_sel)
                    if param_sim > (1.0 - param_th): 
                        is_too_similar = True
                        collision_count += 1
                        break
        
        if not is_too_similar:
            selected.append(cand)
            
    stats = {
        "collision_count": collision_count,
        "avg_jaccard": np.mean(all_jaccards) if all_jaccards else 0.0,
        "collision_rate": collision_count / len(sorted_candidates) if sorted_candidates else 0
    }
    
    return selected, stats
