from __future__ import annotations

from typing import Dict, Iterable, List, Any

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

def predict_direction(model: Any, features: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    Predict probabilities using the trained model.
    
    Args:
        model: Trained sklearn estimator.
        features: List of feature dictionaries.
        
    Returns:
        List of dictionaries mapping Class -> Probability. 
        e.g. {'1': 0.6, '-1': 0.3, '0': 0.1}
    """
    if not features:
        return []

    X = pd.DataFrame(features)
    X.fillna(0.0, inplace=True) # Ensure consistency with training
    
    # Check if model has predict_proba
    if not hasattr(model, "predict_proba"):
        # Fallback or error
        raise ValueError("Model does not support predict_proba")
        
    try:
        # classes_ usually: [-1, 0, 1] or similar sorted integers
        classes = model.classes_
        proba = model.predict_proba(X)
        
        results: List[Dict[str, float]] = []
        for i in range(len(proba)):
            row_probs = proba[i]
            # Map class to probability
            prob_dict = {str(c): float(p) for c, p in zip(classes, row_probs)}
            results.append(prob_dict)
            
        return results
    except Exception as e:
        # Fallback for robustness
        print(f"Prediction failed: {e}")
        # Return uniform or prior?
        return [{str(c): 0.33 for c in model.classes_} for _ in range(len(features))]
