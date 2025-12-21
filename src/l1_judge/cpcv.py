from __future__ import annotations

from typing import Dict, Iterable, List, Generator, Tuple
from itertools import combinations
import numpy as np
import pandas as pd

def compute_cpcv_metrics(perf: Iterable[float]) -> Dict[str, float]:
    """
    Compute CPCV statistics from a list of fold performances (Sharpe or equivalent).
    """
    values = list(perf)
    if not values:
        return {"cpcv_mean": 0.0, "cpcv_worst": 0.0, "cpcv_std": 0.0}
    
    arr = np.array(values)
    mean_v = float(np.mean(arr))
    worst = float(np.min(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    
    # Deflated Sharpe Ratio calculation could go here
    
    return {"cpcv_mean": mean_v, "cpcv_worst": worst, "cpcv_std": std}


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    """
    def __init__(self, n_splits: int = 4, n_test_splits: int = 2, samples_info_sets: pd.Series = None, pct_embargo: float = 0.01):
        """
        Args:
            n_splits (N): Total number of equal groups.
            n_test_splits (k): Number of groups in the test set.
            samples_info_sets: Series with index=event_end_time, value=event_start_time (or vice versa).
                               Used for purging. If None, simple Purging is skipped.
            pct_embargo: Percent of data to embargo after test set.
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo

    def split(self, X: pd.DataFrame, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yields (train_indices, test_indices).
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # 1. Split indices into N groups
        # We assume X is time-sorted
        fold_size = n_samples // self.n_splits
        # remainder goes to last fold
        
        # Define the bounds of each group
        bounds = [i * fold_size for i in range(self.n_splits + 1)]
        bounds[-1] = n_samples
        
        # All combinations of k groups for testing
        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))
        
        for test_groups in test_combinations:
            test_indices_list = []
            for g_idx in test_groups:
                start, end = bounds[g_idx], bounds[g_idx+1]
                test_indices_list.append(indices[start:end])
            
            test_indices = np.concatenate(test_indices_list)
            test_indices.sort()
            
            # Train indices are the rest, MINUS Purge/Embargo
            # Simple version: All non-test indices
            # Advanced version: Remove samples that overlap with test samples (Purging)
            
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_indices] = False
            
            # Apply Purging/Embargo if info provided
            # For this 'Skeleton' implementation, we'll implement simple Gap Embargo
            # Assuming strictly time-sorted.
            
            # Embargo: Drop samples immediately following a test interval
            if self.pct_embargo > 0:
                n_embargo = int(n_samples * self.pct_embargo)
                # Find ends of test blocks
                # Since test_indices are sorted, we look for breaks
                # But we know test groups are chunks.
                for g_idx in test_groups:
                    # After group g_idx ends, embargo n_embargo samples
                    end_pos = bounds[g_idx+1]
                    embargo_end = min(n_samples, end_pos + n_embargo)
                    train_mask[end_pos:embargo_end] = False
            
            train_indices = indices[train_mask]
            
            yield train_indices, test_indices
