from __future__ import annotations

# === WARNING FILTERS FOR WORKER PROCESSES ===
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"joblib.*")
warnings.filterwarnings("ignore", message=".*Series.__setitem__.*")
warnings.filterwarnings("ignore", message=".*physical cores.*")
# Set LOKY CPU count to avoid WinError
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    import multiprocessing
    os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())

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
from src.config import config
from src.templates.registry import TemplateRegistry
from src.features.factory import FeatureFactory
from src.shared.caching import cache # Import cache decorator
from src.features.context_engineering import add_time_features, add_relative_features, add_statistical_features
from src.shared.logger import get_logger

logger = get_logger("orchestration.experiment")

# --- Cached Worker Functions ---

@cache
def _generate_features_cached(df_values: np.ndarray, df_columns: List[str], df_index: Any, genome: Dict[str, Any]) -> pd.DataFrame:
    """
    Cached wrapper for FeatureFactory.
    We pass raw numpy arrays or hashable metadata to ensure joblib hashes it efficiently,
    though passing the DataFrame directly is also supported by joblib.
    Reconstructing DF strictly for compatibility with Factory.
    """
    # Reconstruct DF (Low overhead compared to feature gen)
    df = pd.DataFrame(df_values, columns=df_columns, index=df_index)
    
    factory = FeatureFactory()
    # Ensure column names are lower for factory compatibility
    df.columns = [c.lower() for c in df.columns]
    
    return factory.generate_from_genome(df, genome)

@cache
def _generate_labels_cached(price_values: np.ndarray, price_index: Any, k: float, h: int, vol_span: int) -> pd.DataFrame:
    """
    Cached wrapper for Label Generation.
    """
    prices = pd.Series(price_values, index=price_index)
    return generate_triple_barrier_labels(
        prices=prices,
        k_up=k,
        k_down=k,
        horizon_bars=h,
        vol_span=vol_span
    )

# -------------------------------

def _run_experiment_core(
    policy_spec: PolicySpec,
    df: pd.DataFrame,
    X_features: pd.DataFrame,
    risk_budget: Dict[str, Any],
    include_trade_logic: bool = True,
) -> Dict[str, Any]:
    prices = df["close"]

    k_up = risk_budget.get("k_up", 1.0)
    k_down = risk_budget.get("k_down", 1.0)
    h_param = risk_budget.get("horizon", 20)
    vol_span = 20

    k_param = (k_up + k_down) / 2
    logger.debug(f"[Experiment] Barrier Params: k_up={k_up:.2f}, k_down={k_down:.2f}, horizon={h_param}")

    label_df = _generate_labels_cached(
        price_values=prices.values,
        price_index=prices.index,
        k=k_param,
        h=h_param,
        vol_span=vol_span
    )

    common_idx = X_features.index.intersection(label_df.index)
    valid_mask = label_df.loc[common_idx, "label"].notna()
    final_idx = common_idx[valid_mask]

    if len(final_idx) < 100:
        pass

    X_final_df = X_features.loc[final_idx]
    y_final_series = label_df.loc[final_idx, "label"].astype(int)
    trgt_final_series = label_df.loc[final_idx, "trgt"]

    n_splits = 5
    cv = CombinatorialPurgedKFold(n_splits=n_splits, n_test_splits=1, pct_embargo=0.01)

    oos_preds = []
    oos_indices = []
    oos_scales = []

    lgb_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'seed': 42
    }

    # [Fast/Slow Mode Branching]
    if config.USE_FAST_MODE:
        # Fast Mode: Simple Train/Test Split (No CV, Single Run)
        # Faster evolution, slightly higher risk of overfitting
        split_point = int(len(X_final_df) * 0.8)
        
        # Train on first 80%, Test on last 20%
        train_idx = np.arange(0, split_point)
        test_idx = np.arange(split_point, len(X_final_df))
        
        # Mock iterator for the loop below to run exactly once
        splits = [(train_idx, test_idx)]
    else:
        # Research Mode: Combinatorial Purged K-Fold (5-Fold CV)
        # Robust evaluation, computationally expensive
        cv = CombinatorialPurgedKFold(n_splits=n_splits, n_test_splits=1, pct_embargo=0.01)
        splits = list(cv.split(X_final_df))

    for train_idx, test_idx in splits:
        X_train, y_train = X_final_df.iloc[train_idx], y_final_series.iloc[train_idx]
        X_test = X_final_df.iloc[test_idx]

        guard = MLGuard(params=lgb_params)
        guard.train(features=X_train, targets=y_train)

        # [V6 Autonomy] Dynamic Threshold
        dyn_threshold = policy_spec.tuned_params.get("entry_threshold", config.EVAL_ENTRY_THRESHOLD)
        
        res_df = guard.predict(
            X_test,
            threshold=dyn_threshold,
            max_prob=config.EVAL_ENTRY_MAX_PROB,
        )

        oos_indices.extend(final_idx[test_idx])
        oos_preds.extend(res_df["signal"].tolist())
        oos_scales.extend(res_df["scale"].tolist())

    if not oos_indices:
        cpcv = {"cpcv_mean": 0.0, "cpcv_worst": 0.0, "cpcv_std": 0.0}
        pbo = 1.0
        turnover = 0.0
        trade_logic = {"error": "No OOS results"}
        results_df = pd.DataFrame()
        n_trades = 0
        win_rate = 0.0
        final_guard = None
    else:
        results_df = pd.DataFrame({
            "pred": oos_preds,
            "scale": oos_scales,
        }, index=oos_indices)

        results_df = results_df[~results_df.index.duplicated(keep='first')]
        results_df = results_df.sort_index()

        common_res_idx = results_df.index
        actuals = y_final_series.loc[common_res_idx]
        targs = trgt_final_series.loc[common_res_idx]

        results_df["actual"] = actuals
        results_df["trgt"] = targs

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
        turnover = sum(1 for p in results_df["pred"] if p != 0) / len(results_df)

        res_shift = results_df["pred"].shift(1).fillna(0)
        is_trade_start = (results_df["pred"] != res_shift) & (results_df["pred"] != 0)
        n_trades = int(is_trade_start.sum())

        active_pnl = results_df[results_df["pred"] != 0]["net_pnl"]
        n_winning_bars = (active_pnl > 0).sum()
        win_rate = n_winning_bars / len(active_pnl) if len(active_pnl) > 0 else 0.0

        cpcv["n_trades"] = n_trades
        cpcv["win_rate"] = win_rate
        cpcv["turnover"] = turnover

        if include_trade_logic:
            final_guard = MLGuard(params=lgb_params)
            final_guard.train(features=X_final_df, targets=y_final_series)

            feat_imp = final_guard.get_feature_importance()
            sorted_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))

            top_features = list(sorted_imp.keys())[:3]
            trade_logic = {
                "top_features": top_features,
                "feature_importance": {k: float(v) for k, v in sorted_imp.items()},
                "description": f"Model relied mostly on {', '.join(top_features)} to make decisions."
            }
        else:
            final_guard = None
            trade_logic = {}

    label_hash = f"triple_barrier_k{k_param:.2f}_h{h_param}"
    return {
        "results_df": results_df,
        "cpcv": cpcv,
        "pbo": pbo,
        "turnover": turnover,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "trade_logic": trade_logic,
        "final_guard": final_guard,
        "label_config": {"k_up": k_up, "k_down": k_down, "horizon": h_param},
        "label_hash": label_hash,
    }


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()

def _label_imbalance(labels: List[int]) -> bool:
    if not labels:
        return True
    counts = {label: labels.count(label) / len(labels) for label in set(labels)}
    return any(freq < 0.1 for freq in counts.values())


def build_record_and_artifact(
    policy_spec: PolicySpec,
    df: pd.DataFrame,
    X_features: pd.DataFrame,
    core: Dict[str, Any],
    scorecard: Optional[Dict[str, Any]] = None,
    module_key: Optional[str] = None,
    eval_stage: Optional[str] = None,
) -> tuple[LedgerRecord, ArtifactBundle]:
    results_df = core.get("results_df")
    cpcv = dict(core.get("cpcv") or {})
    pbo = core.get("pbo", 0.0)
    trade_logic = core.get("trade_logic") or {}
    final_guard = core.get("final_guard")
    label_hash = core.get("label_hash", "")
    label_config = core.get("label_config") or {}

    if scorecard:
        cpcv.update(scorecard)
    if module_key:
        cpcv["module_key"] = module_key
    if eval_stage:
        cpcv["eval_stage"] = eval_stage

    if results_df is None or results_df.empty:
        backtest_results_dict = []
    else:
        backtest_results_dict = results_df.reset_index().rename(columns={"index": "date"}).to_dict(orient="records")

    verdict_dump = trade_logic
    if module_key or scorecard or eval_stage:
        verdict_dump = {
            "trade_logic": trade_logic,
            "module_key": module_key,
            "eval": scorecard,
            "stage": eval_stage,
        }

    hard_fail = []
    if cpcv.get("cpcv_worst", 0.0) < -1.5:
        hard_fail.append("CPCV_WORST_TOO_LOW")
    if pbo > 0.4:
        hard_fail.append("PBO_TOO_HIGH")

    exp_id = str(uuid4())

    artifact = ArtifactBundle(
        label_config=label_config,
        direction_model=final_guard,
        risk_model=None,
        calibration_metrics={},
        metadata={"genome": policy_spec.feature_genome, "risk_budget": policy_spec.risk_budget},
        backtest_results=backtest_results_dict
    )

    ledger_record = LedgerRecord(
        exp_id=exp_id,
        timestamp=time.time(),
        policy_spec=policy_spec,
        data_hash=_hash_payload(df.values.tobytes()),
        feature_hash=_hash_payload(X_features.columns.tolist()),
        label_hash=label_hash,
        model_artifact_ref="",
        cpcv_metrics=cpcv,
        pbo=pbo,
        risk_report={"ok": True},
        reason_codes=hard_fail,
        verdict_dump=verdict_dump
    )

    return ledger_record, artifact

def run_experiment(
    registry: TemplateRegistry, # Legacy arg, kept for compatibility signature
    policy_spec: PolicySpec,
    market_data: pd.DataFrame,
    ledger_repo: Optional[LedgerRepo] = None,
) -> tuple[LedgerRecord, ArtifactBundle]:
    
    # 0. Check Dependencies
    if lgb is None:
        raise ImportError("LightGBM must be installed")
    
    # 0. Dependencies & Config (Worker Process)
    import logging
    logging.getLogger("l2.ml_guard").setLevel(logging.WARNING)
    logging.getLogger("feature.registry").setLevel(logging.WARNING)
    logging.getLogger("feature.custom_loader").setLevel(logging.WARNING)
    
    # 0. Validation
    if market_data.empty:
        raise ValueError("Market data is empty")
        
    df = market_data.copy()
    df.columns = [c.lower() for c in df.columns]

    # 1. Feature Generation (Genome -> Features)
    # Use Cached Version
    # Pass numpy array to speed up hashing or standard DF. Joblib handles DF efficiently.
    # We pass strict args to match the cached function signature.
    X_features = _generate_features_cached(
        df_values=df.values,
        df_columns=df.columns.tolist(),
        df_index=df.index,
        genome=policy_spec.feature_genome
    )
    
    if X_features.empty:
        raise ValueError("No features generated from genome. Check definitions.")

    # [V3 Update] Automatically Merge Macro/Context Data
    # Identify non-OHLCV columns in the original market data (Macro assets, Indices, etc.)
    # These are already in 'df' (market_data.copy)
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    macro_cols = [c for c in df.columns if c not in ohlcv_cols]
    
    if macro_cols:
        # [V5 Revert] We do NOT force merge. We rely on the Genome to pick 'MACRO_VIX' etc.
        # However, we MUST ensure the factory has access to these columns.
        # The '_generate_features_cached' passes the FULL 'df' (which includes macro) to the factory.
        # So we just don't do anything here. The factory snippets will pull what they need.
        logger.debug(f"[Experiment] Available macro columns in DF: {macro_cols}")

    core = _run_experiment_core(
        policy_spec=policy_spec,
        df=df,
        X_features=X_features,
        risk_budget=policy_spec.risk_budget or {},
        include_trade_logic=True,
    )

    scorecard = {
        **(core.get("cpcv") or {}),
        "pbo": core.get("pbo", 0.0),
        "turnover": core.get("turnover", 0.0),
    }

    ledger_record, artifact = build_record_and_artifact(
        policy_spec=policy_spec,
        df=df,
        X_features=X_features,
        core=core,
        scorecard=scorecard,
    )

    return ledger_record, artifact
