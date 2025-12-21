
from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Optional, Any
from uuid import uuid4
import pandas as pd
import numpy as np
try:
    import lightgbm as lgb
except ImportError:
    # Fallback or strict error
    lgb = None
try:
    import shap
except ImportError:
    shap = None

from src.contracts import FixSuggestion, Forecast, LedgerRecord, PolicySpec, Verdict
from src.l1_judge.cpcv import compute_cpcv_metrics, CombinatorialPurgedKFold
from src.l1_judge.fix_suggester import generate_fix_suggestion
from src.l1_judge.pbo import compute_pbo
from src.l1_judge.risk_engine import assess_risk
from src.l2_sl.artifacts import ArtifactBundle
from src.l2_sl.calibration.calibrate import calibrate_probabilities
from src.l2_sl.labeling.vol_scaling import generate_triple_barrier_labels
from src.l2_sl.ml_guard import MLGuard
from src.ledger.repo import LedgerRepo
from src.templates.registry import TemplateRegistry
from src.features.factory import FeatureFactory


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()

def _label_imbalance(labels: List[int]) -> bool:
    if not labels:
        return True
    counts = {label: labels.count(label) / len(labels) for label in set(labels)}
    return any(freq < 0.1 for freq in counts.values())

def run_experiment(
    registry: TemplateRegistry, # Legacy arg, kept for compatibility signature
    policy_spec: PolicySpec,
    market_data: pd.DataFrame,
    ledger_repo: Optional[LedgerRepo] = None,
) -> LedgerRecord:
    
    # 0. Check Dependencies
    if lgb is None:
        raise ImportError("LightGBM must be installed")
    
    # 0. Validation
    if market_data.empty:
        raise ValueError("Market data is empty")
        
    df = market_data.copy()
    df.columns = [c.lower() for c in df.columns]

    # 1. Feature Generation (Genome -> Features)
    feature_factory = FeatureFactory()
    X_features = feature_factory.generate_from_genome(df, policy_spec.feature_genome)
    
    if X_features.empty:
        raise ValueError("No features generated from genome. Check definitions.")

    # 2. Labeling (Fixed for now: Triple Barrier)
    prices = df["close"]
    k_param = 1.0 # Default
    h_param = 20 # Default
    vol_span = 20
    
    label_df = generate_triple_barrier_labels(
        prices=prices,
        k_up=k_param,
        k_down=k_param,
        horizon_bars=h_param,
        vol_span=vol_span
    )
    
    # 3. Alignment
    common_idx = X_features.index.intersection(label_df.index)
    valid_mask = label_df.loc[common_idx, "label"].notna()
    final_idx = common_idx[valid_mask]
    
    if len(final_idx) < 100:
        pass # Better handling needed

    X_final_df = X_features.loc[final_idx]
    y_final_series = label_df.loc[final_idx, "label"].astype(int)
    trgt_final_series = label_df.loc[final_idx, "trgt"]
    
    # 4. Instant ML Training (LightGBM)
    
    n_splits = 5
    cv = CombinatorialPurgedKFold(n_splits=n_splits, n_test_splits=1, pct_embargo=0.01)
    
    oos_preds = []
    oos_indices = []
    
    # Model Params (Fixed or derived from Genome later?)
    lgb_params = {
        'objective': 'multiclass',
        'num_class': 3, # -1, 0, 1 mapped to 0, 1, 2 internally by LGB usually needed
        'metric': 'multi_logloss', # or 'multi_error' for accuracy
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        # 'force_col_wise': True, # Optimization
        'seed': 42 # Reproducibility
    }
    
    # Map class -1, 0, 1 to 0, 1, 2 for LGB is now handled by MLGuard internally if objective is multiclass
    
    for train_idx, test_idx in cv.split(X_final_df):
        X_train, y_train = X_final_df.iloc[train_idx], y_final_series.iloc[train_idx]
        X_test = X_final_df.iloc[test_idx]
        
        # Instantiate MLGuard
        guard = MLGuard(params=lgb_params)
        
        # Train (Pass original labels, MLGuard handles mapping)
        guard.train(features=X_train, targets=y_train)
        
        # Predict
        res_df = guard.predict(X_test, threshold=0.7, max_prob=0.9)
        
        # Collect Results
        oos_indices.extend(final_idx[test_idx])
        oos_preds.extend(res_df["signal"].tolist())
        
        # We also need to store scale. 
        # But oos_preds is a flat list. We need a parallel list for scale.
        if "oos_scales" not in locals(): oos_scales = []
        oos_scales.extend(res_df["scale"].tolist())

    # 5. Compile Results
    if not oos_indices:
        # Failure case
        cpcv = {"cpcv_mean": 0.0, "cpcv_worst": 0.0}
        pbo = 1.0
        turnover = 0.0
        trade_logic = {"error": "No OOS results"}
        backtest_results_dict = []
    else:
        results_df = pd.DataFrame({
            "pred": oos_preds,
            "scale": oos_scales,
        }, index=oos_indices)
        
        # De-duplicate and Sort
        results_df = results_df[~results_df.index.duplicated(keep='first')]
        results_df = results_df.sort_index()
        
        common_res_idx = results_df.index
        actuals = y_final_series.loc[common_res_idx]
        targs = trgt_final_series.loc[common_res_idx]
        
        results_df["actual"] = actuals
        results_df["trgt"] = targs
        
        # PnL Calc
        cost_bps = policy_spec.execution_assumption.get("cost_bps", 5)
        pnl = []
        for _, row in results_df.iterrows():
            pred_s = row["pred"]
            act_s = row["actual"]
            trgt = row["trgt"]
            
            if pred_s == 0:
                pnl.append(0.0)
            else:
                scale = row.get("scale", 1.0)
                gross = (1.0 if pred_s == act_s else -1.0) * scale
                impact = ((2 * cost_bps / 10000.0) / (trgt + 1e-6)) * scale
                pnl.append(gross - impact)
        results_df["net_pnl"] = pnl
        
        cpcv = compute_cpcv_metrics(pnl)
        pbo = compute_pbo([pnl], target_idx=0)
        turnover = sum(1 for p in results_df["pred"] if p!=0) / len(results_df)

        # Advanced Trade Stats
        res_shift = results_df["pred"].shift(1).fillna(0)
        # A trade starts when prediction changes to a non-zero value
        is_trade_start = (results_df["pred"] != res_shift) & (results_df["pred"] != 0)
        n_trades = int(is_trade_start.sum())

        # Win Rate (Bar-based)
        active_pnl = results_df[results_df["pred"] != 0]["net_pnl"]
        n_winning_bars = (active_pnl > 0).sum()
        win_rate = n_winning_bars / len(active_pnl) if len(active_pnl) > 0 else 0.0
        
        # Inject into cpcv metrics for persistence
        cpcv["n_trades"] = n_trades
        cpcv["win_rate"] = win_rate
        cpcv["turnover"] = turnover # Exposure actually
        
        # Backtest Dict
        backtest_results_dict = results_df.reset_index().rename(columns={"index": "date"}).to_dict(orient="records")
        
        # Explainability (Aggregating global feature importance)
        # Train one final model on full data for global importance
        final_guard = MLGuard(params=lgb_params)
        final_guard.train(features=X_final_df, targets=y_final_series)
        
        feat_imp = final_guard.get_feature_importance()
        # Sort by importance
        sorted_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
        
        # Top 3 Logic
        top_features = list(sorted_imp.keys())[:3]
        trade_logic = {
            "top_features": top_features,
            "feature_importance": {k: float(v) for k, v in sorted_imp.items()},
            "description": f"Model relied mostly on {', '.join(top_features)} to make decisions."
        }

    # 6. Verdict and Ledger
    hard_fail = []
    if cpcv["cpcv_worst"] < -1.5: hard_fail.append("CPCV_WORST_TOO_LOW")
    if pbo > 0.4: hard_fail.append("PBO_TOO_HIGH")
    # if turnover...
    
    scorecard = {**cpcv, "pbo": pbo, "turnover": turnover}
    
    exp_id = str(uuid4())
    
    # Artifact creation (Basic for now)
    artifact = ArtifactBundle(
        label_config={"k": 1.0, "H": 20},
        direction_model=final_guard, # Persist the trained MLGuard model
        risk_model=None,
        calibration_metrics={},
        metadata={"genome": policy_spec.feature_genome},
        backtest_results=backtest_results_dict
    )
    
    ledger_record = LedgerRecord(
        exp_id=exp_id,
        timestamp=time.time(),
        policy_spec=policy_spec,
        data_hash=_hash_payload(df.values.tobytes()),
        feature_hash=_hash_payload(X_features.columns.tolist()),
        label_hash="manual_triple_barrier",
        model_artifact_ref="",
        cpcv_metrics=cpcv,
        pbo=pbo,
        risk_report={"ok": True}, 
        reason_codes=hard_fail,
        verdict_dump=trade_logic 
    )
    
    if ledger_repo:
         ledger_repo.save_record(ledger_record, artifact)
        
    return ledger_record
