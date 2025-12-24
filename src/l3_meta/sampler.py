from typing import List, Dict, Any, Optional
import random
from src.contracts import PolicySpec
from src.l1_judge.diversity import calculate_genome_similarity
from src.shared.logger import get_logger

logger = get_logger("l3.sampler")

class DiversitySampler:
    """
    [V15] Diversity-aware Sampling
    Ensures candidates are diverse even before evaluation.
    """
    def __init__(self, jaccard_threshold: float = 0.8):
        self.jaccard_threshold = jaccard_threshold

    def filter_diverse(self, candidates: List[PolicySpec], existing: List[PolicySpec]) -> List[PolicySpec]:
        """Filters out candidates that are too similar to existing or each other."""
        selected = []
        pool = existing + selected
        
        for cand in candidates:
            is_redundant = False
            for target in pool:
                sim = calculate_genome_similarity(cand, target)
                if sim > self.jaccard_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected.append(cand)
                pool.append(cand)
                
        return selected

def sample_with_diversity(
    propose_fn, 
    n_jobs: int, 
    regime, 
    history, 
    jaccard_th: float = 0.8,
    max_attempts: int = 5
) -> List[PolicySpec]:
    """
    [V15] Batch sampling with iterative diversification.
    """
    final_policies = []
    attempts = 0
    
    sampler = DiversitySampler(jaccard_threshold=jaccard_th)
    
    while len(final_policies) < n_jobs and attempts < max_attempts:
        needed = n_jobs - len(final_policies)
        # Generate slightly more than needed to allow for filtering
        raw_candidates = [propose_fn(regime, history) for _ in range(needed * 2)]
        
        diverse_batch = sampler.filter_diverse(raw_candidates, final_policies)
        final_policies.extend(diverse_batch)
        attempts += 1
        
    if len(final_policies) < n_jobs:
        logger.warning(f"Only managed to find {len(final_policies)} diverse policies after {attempts} attempts.")
        # Fill rest with regular proposals if we must
        needed = n_jobs - len(final_policies)
        final_policies.extend([propose_fn(regime, history) for _ in range(needed)])
        
    return final_policies[:n_jobs]
