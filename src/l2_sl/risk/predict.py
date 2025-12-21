from __future__ import annotations

from typing import Dict, Iterable, List, Any
import numpy as np

def predict_risk(model: Dict[str, Any], current_vols: Iterable[float]) -> List[float]:
    """
    Predict/Score risk based on current volatility and trained thresholds.
    Returns a 'Tail Risk Score' (0 to 1 scaling or Z-score like).
    
    If current_vol > p95, score grows high.
    """
    p95 = model.get("p95", 1.0)
    avg = model.get("avg_vol", 0.0)
    
    scores = []
    for v in current_vols:
        # Simple heuristic: Ratio to 95th percentile
        # If > 1.0, it's tail risk.
        if p95 > 0:
            score = v / p95
        else:
            score = 0.0
        scores.append(score)
        
    return scores
