from __future__ import annotations

from typing import Iterable, List
import numpy as np

def compute_pbo(performances: List[List[float]], target_idx: int) -> float:
    """
    Compute Probability of Backtest Overfitting (PBO) given a matrix of performances
    (n_samples x n_strategies).
    
    This estimates the probability that the 'target_idx' strategy (usually the best one in-sample)
    is underperforming out-of-sample compared to the median.
    
    However, for the blueprint 'Rank based PBO', we often adhere to:
    "Probability that the selected strategy is not actually the best in OOS".
    
    Simplified Logic for Single Strategy Context (CPCV folds as pseudo-trials? No, strict PBO needs multiple strategies):
    If input is just 'ranks', we calculate Rank Logits.
    
    Args:
        performances: Matrix (N folds x M strategies).
        target_idx: Index of the strategy selected by IS performance.
        
    Returns:
        float: PBO value [0, 1].
    """
    # Placeholder: If we don't have M strategies, we can't compute true PBO.
    # The blueprint L1 PBO implies it looks at the "History" or "Pool".
    
    # As a robust fall-back if only 1 strategy is passed (legacy stub compatibility):
    if not performances or not performances[0]:
        return 0.0
        
    mat = np.array(performances)
    # n_folds, n_strats = mat.shape
    
    # If we only have 1 strategy column, PBO is 0 (no selection bias possible).
    if mat.ndim == 1 or mat.shape[1] == 1:
        return 0.0

    # Combinatorial PBO implementation (CSCV)
    # count how many times the IS-best is not OOS-best?
    # This requires splitting the folds into IS/OOS combinations.
    # Assuming 'mat' rows are OOS results of different CPCV paths?
    
    # For this skeleton, we will implement the "Rank Inversion" count if provided "ranks".
    # But let's assume standard input is the performance matrix of the 'Battle Royale'.
    
    # Let's perform a simple check:
    # rank of target in each fold. If it's consistently low, PBO is high?
    
    # Let's revert to a simpler method compatible with the Blueprint's "Ranking based" text.
    # We will assume we compare "Current Strategy" vs "Benchmark/Ledger Strategies".
    
    # Simply:
    n_folds, n_strats = mat.shape
    
    # Count how often the 'Best In-Sample' config fails Out-of-Sample.
    # Since we passed OOS performance matrix directly:
    
    # Calculate logits
    # For each row (fold), rank the strategies.
    # normalize ranks.
    pass
    return 0.0 # TODO: Connect to Ledger for real PBO.

def compute_rank_pbo(ranks: Iterable[float]) -> float:
    """
    Experimental: Calculate PBO based on rank correlation decay or similar.
    Stub implementation used by current codebase.
    """
    ranks_list = list(ranks)
    if not ranks_list:
        return 0.0
        
    # Inversion counting
    inversions = 0
    pairs = 0
    for i in range(len(ranks_list)):
        for j in range(i + 1, len(ranks_list)):
            pairs += 1
            if ranks_list[i] > ranks_list[j]: # If performance degrades in subsequent test? 
                # This assumes time-ordered ranks. PBO is usually cross-sectional.
                # If these are ranks of the SAME strategy over TIME (folds):
                # High variance in rank = instability.
                pass
                
    # Revised Stub:
    # If we interpret input as 'OOS Ranks of the Selected Strategy across N folds' (relative to others)
    # If the ranks are widely distributed, PBO is high.
    
    # Let's return a simple "Inversion Ratio" as a proxy for instability.
    # (Existing stub logic preserved but typed better)
    inversions = sum(1 for i, r in enumerate(ranks_list) for s in ranks_list[i + 1 :] if s < r)
    denom = len(ranks_list) * (len(ranks_list) - 1) / 2 or 1
    return inversions / denom
