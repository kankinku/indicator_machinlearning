import numpy as np
import pandas as pd
from typing import List, Dict, Any

class FastFilter:
    """
    [V14] Fast Filter
    Vectorized, high-speed scoring of trading strategies.
    Used for Stage 1 screening to identify top candidates for detailed evaluation.
    """
    def __init__(self):
        pass

    def score_batch(
        self, 
        df: pd.DataFrame, 
        signals_list: List[pd.Series], 
        risk_budgets: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Calculates a 'fast score' for a batch of strategies using vectorization.
        """
        if not signals_list:
            return []

        # Vectorized calculation for each signal
        if "close" not in df.columns:
            # Fallback if close is missing or case-insensitive
            close_col = next((c for c in df.columns if c.lower() == "close"), None)
            if close_col:
                close = df[close_col].values
            else:
                return [-99.0] * len(signals_list)
        else:
            close = df["close"].values
            
        if len(close) < 2:
            return [-99.0] * len(signals_list)
            
        returns = np.diff(close) / close[:-1]
        target_len = len(returns)
        
        scores = []
        for i, signals in enumerate(signals_list):
            if signals.empty:
                scores.append(-99.0)
                continue
                
            sig_values = signals.values
            if len(sig_values) > target_len:
                sig_values = sig_values[:target_len]
            elif len(sig_values) < target_len:
                # Pad with zeros or truncate returns? 
                # Better to just skip/penalty if they don't match
                scores.append(-99.0)
                continue
                
            # strat_returns = sig_values * returns (if they match)
            # Actually, sig_values should be shifted by 1 if it represents entry signal at time t
            # and we want return from t to t+1. 
            # In our system, sig[t] means entry at close of t, so return is returns[t+1]
            # Wait, returns[i] is (close[i+1]-close[i])/close[i].
            # So sig[i] matches returns[i] if it means entry at close of i.
            
            try:
                strat_returns = sig_values * returns
                total_ret = np.sum(strat_returns)
                n_trades = np.count_nonzero(sig_values)
                
                if n_trades == 0:
                    scores.append(-99.0)
                    continue
                    
                vol = np.std(strat_returns)
                sharpe = (np.mean(strat_returns) / vol) if vol > 1e-9 else 0.0
                score = total_ret * 1000.0 + sharpe * 10.0
                scores.append(float(score))
            except Exception:
                scores.append(-99.0)
            
        return scores
