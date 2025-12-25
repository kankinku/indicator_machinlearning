from typing import List, Tuple
from src.contracts import PolicySpec

class DualStageEvaluator:
    """
    [V14] Dual Stage Evaluator
    Manages the transition between Fast Filter and Detailed Evaluation.
    """
    def __init__(self):
        pass

    def select_mixed_candidates(
        self, 
        scored_policies: List[Tuple[float, PolicySpec]], 
        elite_n: int = 5,
        explorer_n: int = 5
    ) -> List[PolicySpec]:
        """
        Selects a mix of top-performing ('elite') and randomly selected ('explorer') strategies.
        """
        if not scored_policies:
            return []
            
        # Sort by score descending
        sorted_policies = sorted(scored_policies, key=lambda x: x[0], reverse=True)
        
        # 1. Take Elites
        elites = [p for s, p in sorted_policies[:elite_n]]
        
        # 2. Take Explorers (diversity-aware among the rest)
        rest = sorted_policies[elite_n:]
        import random
        explorer_count = min(len(rest), explorer_n)
        if explorer_count <= 0:
            return elites

        if elites and rest:
            try:
                from src.l1_judge.diversity import calculate_genome_similarity
                scored = []
                for _, cand in rest:
                    max_sim = max(
                        calculate_genome_similarity(cand, elite) for elite in elites
                    )
                    diversity_score = 1.0 - max_sim
                    scored.append((diversity_score, cand))
                scored.sort(key=lambda x: x[0], reverse=True)
                explorers = [p for _, p in scored[:explorer_count]]
            except Exception:
                explorers = [p for s, p in random.sample(rest, explorer_count)] if rest else []
        else:
            explorers = [p for s, p in random.sample(rest, explorer_count)] if rest else []
        
        return elites + explorers
