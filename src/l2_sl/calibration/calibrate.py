from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import brier_score_loss

def calibrate_probabilities(probs: List[Dict[str, float]], labels: List[int]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Calibrate probabilities and calculate ECE/Brier Score.
    
    Args:
        probs: List of dicts {class: prob}.
        labels: True labels corresponding to the probs.
    
    Returns:
        calibrated_probs: (Currently just normalized input as strict Isotonic requires fitting)
        metrics: {'ece': float, 'brier': float}
    """
    # 1. Normalize
    calibrated: List[Dict[str, float]] = []
    # Convert to list of arrays for metric calc
    y_true = []
    y_prob_positive = []
    
    # Assume binary or 3-class. Let's focus on class '1' (Up) for metrics
    target_cls = '1'
    
    for i, prob in enumerate(probs):
        total = sum(prob.values()) or 1.0
        norm_prob = {k: v / total for k, v in prob.items()}
        calibrated.append(norm_prob)
        
        if i < len(labels):
            y_true.append(1 if str(labels[i]) == '1' else 0)
            y_prob_positive.append(norm_prob.get('1', 0.0))

    # 2. Compute Metrics (Brier)
    if not y_true:
        return calibrated, {"ece": 0.0, "brier": 0.0}
        
    brier = brier_score_loss(y_true, y_prob_positive)
    
    # Simple ECE approximation
    # binning
    y_prob_arr = np.array(y_prob_positive)
    y_true_arr = np.array(y_true)
    
    # 5 bins
    bins = np.linspace(0, 1, 6)
    ece = 0.0
    for j in range(len(bins)-1):
        mask = (y_prob_arr >= bins[j]) & (y_prob_arr < bins[j+1])
        if np.any(mask):
            bin_conf = np.mean(y_prob_arr[mask])
            bin_acc = np.mean(y_true_arr[mask])
            ece += np.abs(bin_conf - bin_acc) * (np.sum(mask) / len(y_true))
            
    return calibrated, {"ece": float(ece), "brier": float(brier)}
