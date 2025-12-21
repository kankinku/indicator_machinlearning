from __future__ import annotations

from typing import Dict, Iterable, List, Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator

def train_direction_model(features: List[Dict[str, float]], labels: Iterable[int]) -> Any:
    """
    Train a GBDT model for direction prediction.
    
    Args:
        features: List of feature dictionaries (records).
        labels: Iterable of integer labels (1, -1, 0).
        
    Returns:
        Trained sklearn model object.
    """
    # Convert to DataFrame
    X = pd.DataFrame(features)
    y = list(labels)
    
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Empty training data")
        
    # Standardize or fill missing? GBDT handles some, but good practice to fill.
    X.fillna(0.0, inplace=True)
    
    # Initialize basic GBDT
    # In a real meta-optimization system, hyperparameters would be passed in.
    # For now, we use robust defaults or what the Template specified (but this function signature doesn't take params yet).
    # We'll assume default config for this "Basic" implementation.
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X, y)
    
    return clf
