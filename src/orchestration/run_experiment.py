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
    lgb = None
try:
    import shap
except ImportError:
    shap = None

from src.contracts import FixSuggestion, Forecast, LedgerRecord, PolicySpec, Verdict
from src.l1_judge.cpcv import compute_cpcv_metrics, CombinatorialPurgedKFold
from src.l1_judge.fix_suggester import generate_fix_suggestion
from src.l1_judge.evaluator import validate_sample, SampleMetrics # [vNext] Validation
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
from src.shared.caching import cache, get_feature_cache, get_label_cache
from src.shared.backtest import run_signal_backtest, build_signal_series, derive_trade_params
from src.features.context_engineering import add_time_features, add_relative_features, add_statistical_features
from src.shared.logger import get_logger
import ta

logger = get_logger("orchestration.experiment")

# --- 캐시 인스턴스 ---
_feature_mem_cache = get_feature_cache()
_label_mem_cache = get_label_cache()

# --- Cached Worker Functions ---

def _generate_features_cached(df_values: np.ndarray, df_columns: List[str], df_index: Any, genome: Dict[str, Any]) -> pd.DataFrame:
    """
    2-Tier Cached Feature Generation.
    """
    df_temp = pd.DataFrame(df_values, columns=df_columns, index=df_index)
    cache_key = _feature_mem_cache.make_key(df_temp, genome)
    
    cached_result = _feature_mem_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    result = _generate_features_disk_cached(df_values, df_columns, df_index, genome)
    _feature_mem_cache.set(cache_key, result)
    
    return result


@cache
def _generate_features_disk_cached(df_values: np.ndarray, df_columns: List[str], df_index: Any, genome: Dict[str, Any]) -> pd.DataFrame:
    """디스크 캐시를 사용하는 피처 생성 함수."""
    df = pd.DataFrame(df_values, columns=df_columns, index=df_index)
    factory = FeatureFactory()
    df.columns = [c.lower() for c in df.columns]
    return factory.generate_from_genome(df, genome)


@cache
def _generate_labels_cached(price_values: np.ndarray, price_index: Any, k: float, h: int, vol_span: int) -> pd.DataFrame:
    """Cached wrapper for Label Generation."""
    prices = pd.Series(price_values, index=price_index)
    labels = generate_triple_barrier_labels(
        prices=prices,
        k_up=k,
        k_down=k,
        horizon_bars=h,
        vol_span=vol_span
    )
    # [V8.4] Long-Only 변환: Sell(-1) -> Hold(0)
    labels['label'] = labels['label'].replace(-1, 0)
    return labels


def _run_experiment_core(
    policy_spec: PolicySpec,
    df: pd.DataFrame,
    X_features: pd.DataFrame,
    risk_budget: Dict[str, Any],
    include_trade_logic: bool = True,
) -> Dict[str, Any]:
    """
    핵심 실험 실행 함수.
    
    [V9 Major Refactor] Single Source of Truth
    - 모든 평가 지표는 run_signal_backtest 결과에서만 추출
    - 라벨 일치 기반 평가 완전 제거
    """
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

    if len(final_idx) < 50:
        logger.warning(f"[Experiment] Insufficient data: {len(final_idx)} samples")
        return _empty_core_result(k_up, k_down, h_param, k_param)

    X_final_df = X_features.loc[final_idx]
    y_final_series = label_df.loc[final_idx, "label"].astype(int)
    trgt_final_series = label_df.loc[final_idx, "trgt"]

    t_train_start = time.time()

    # LightGBM 파라미터 설정
    if config.USE_FAST_MODE:
        lgb_params = {
            'objective': 'binary',
            'num_class': 1,
            'metric': 'binary_logloss',
            'is_unbalance': True,  # [V9.1] Encourage trading
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': 15,
            'num_leaves': 7,
            'learning_rate': 0.10,
            'min_child_samples': 40,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 0.2,
            'lambda_l2': 0.2,
            'seed': 42
        }
    else:
        lgb_params = {
            'objective': 'binary',
            'num_class': 1,
            'metric': 'binary_logloss',
            'is_unbalance': True,  # [V9.1] Encourage trading
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': 100,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'seed': 42
        }

    # Walk-Forward Split with Purge
    purge_bars = max(10, h_param)  # Triple Barrier horizon 고려
    split_point = int(len(X_final_df) * 0.7)  # 70/30 분할
    
    # Purge: 학습 데이터 끝에서 purge_bars 만큼 제외
    train_end = max(50, split_point - purge_bars)
    train_idx = np.arange(0, train_end)
    test_idx = np.arange(split_point, len(X_final_df))
    
    if len(test_idx) < 30:
        logger.warning(f"[Experiment] Test set too small: {len(test_idx)}")
        return _empty_core_result(k_up, k_down, h_param, k_param)

    splits = [(train_idx, test_idx)]

    oos_preds = []
    oos_indices = []
    oos_scales = []
    oos_probs = []

    for train_idx, test_idx in splits:
        X_train = X_final_df.iloc[train_idx]
        y_train = y_final_series.iloc[train_idx]
        X_test = X_final_df.iloc[test_idx]

        guard = MLGuard(params=lgb_params)
        guard.train(features=X_train, targets=y_train)

        # [V9] Entry Threshold 대폭 완화 - 더 많은 신호 생성
        dyn_threshold = policy_spec.tuned_params.get("entry_threshold", config.EVAL_ENTRY_THRESHOLD)
        
        res_df = guard.predict(
            X_test,
            threshold=dyn_threshold,
            max_prob=config.EVAL_ENTRY_MAX_PROB,
        )

        oos_indices.extend(final_idx[test_idx])
        oos_preds.extend(res_df["signal"].tolist())
        oos_scales.extend(res_df["scale"].tolist())
        oos_probs.extend(res_df["raw_prob"].tolist())
    
    t_train_end = time.time()
    t_train = t_train_end - t_train_start

    if not oos_indices:
        return _empty_core_result(k_up, k_down, h_param, k_param)

    # OOS 결과 DataFrame 생성
    results_df = pd.DataFrame({
        "pred": oos_preds,
        "scale": oos_scales,
        "prob": oos_probs,
    }, index=oos_indices)

    results_df = results_df[~results_df.index.duplicated(keep='first')]
    results_df = results_df.sort_index()

    # =====================================================
    # [V9 CORE CHANGE] Single Source of Truth: Backtest
    # 라벨 일치 기반 평가 완전 제거
    # 실제 가격 기반 Backtest만 사용
    # =====================================================
    
    # OOS 구간의 가격 데이터 추출
    oos_start = results_df.index.min()
    oos_end = results_df.index.max()
    oos_price_df = df.loc[oos_start:oos_end].copy()
    
    cost_bps = policy_spec.execution_assumption.get("cost_bps", config.DEFAULT_COST_BPS)
    
    # 실제 가격 기반 Backtest 실행
    bt_result = run_signal_backtest(
        price_df=oos_price_df,
        results_df=results_df,
        risk_budget=risk_budget,
        cost_bps=cost_bps,
    )
    
    # Backtest 결과에서 모든 지표 추출
    n_trades = bt_result.trade_count
    win_rate = bt_result.win_rate
    total_return_pct = bt_result.total_return_pct
    mdd_pct = bt_result.mdd_pct
    
    # 거래별 수익률로 CPCV 계산 (Backtest 기반)
    trade_returns = [t["return_pct"] / 100.0 for t in bt_result.trades]
    if trade_returns:
        cpcv = compute_cpcv_metrics(trade_returns)
    else:
        cpcv = {"cpcv_mean": 0.0, "cpcv_worst": 0.0, "cpcv_std": 0.0}
    
    # PBO 계산
    pbo = compute_pbo([trade_returns], target_idx=0) if trade_returns else 1.0
    
    # 턴오버 계산
    turnover = n_trades / len(results_df) if len(results_df) > 0 else 0.0
    
    # [V9] CPCV metrics에 Backtest 결과 저장
    cpcv["n_trades"] = n_trades
    cpcv["win_rate"] = win_rate
    cpcv["total_return_pct"] = total_return_pct
    cpcv["mdd_pct"] = mdd_pct
    cpcv["turnover"] = turnover
    cpcv["oos_bars"] = len(results_df)
    cpcv["signal_count"] = int((results_df["pred"] != 0).sum())

    # [vNext] Exposure Ratio Calculation (Anti-Zero Exposure)
    exposure_ratio = 0.0
    if len(results_df) > 0:
        exposure_ratio = float((results_df["pred"] != 0).sum()) / len(results_df)
    cpcv["exposure_ratio"] = exposure_ratio

    # Trade Logic 추출
    trade_logic = {}
    final_guard = None
    
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

    label_hash = f"triple_barrier_k{k_param:.2f}_h{h_param}"
    
    return {
        "results_df": results_df,
        "cpcv": cpcv,
        "pbo": pbo,
        "turnover": turnover,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "total_return_pct": total_return_pct,
        "mdd_pct": mdd_pct,
        "trade_logic": trade_logic,
        "final_guard": final_guard,
        "label_config": {"k_up": k_up, "k_down": k_down, "horizon": h_param},
        "label_hash": label_hash,
        "t_train": t_train,
        "bt_result": bt_result,  # Backtest 결과 전체 저장
    }


def _empty_core_result(k_up: float, k_down: float, h_param: int, k_param: float) -> Dict[str, Any]:
    """빈 결과 반환 헬퍼"""
    return {
        "results_df": pd.DataFrame(),
        "cpcv": {"cpcv_mean": 0.0, "cpcv_worst": 0.0, "cpcv_std": 0.0, "n_trades": 0, "win_rate": 0.0},
        "pbo": 1.0,
        "turnover": 0.0,
        "n_trades": 0,
        "win_rate": 0.0,
        "total_return_pct": 0.0,
        "mdd_pct": 0.0,
        "trade_logic": {"error": "Insufficient data"},
        "final_guard": None,
        "label_config": {"k_up": k_up, "k_down": k_down, "horizon": h_param},
        "label_hash": f"triple_barrier_k{k_param:.2f}_h{h_param}",
        "t_train": 0.0,
        "bt_result": None,
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

    # [V9] Validation based on Backtest results
    # [vNext] Validation Layer Integration (Hard Gates)
    hard_fail = []
    
    # Convert dict to SampleMetrics for validation
    metrics_obj = SampleMetrics(
        total_return_pct=cpcv.get("total_return_pct", 0.0),
        mdd_pct=cpcv.get("mdd_pct", 0.0),
        reward_risk=cpcv.get("reward_risk", 0.0),
        vol_pct=0.0,
        trade_count=cpcv.get("n_trades", 0),
        valid_trade_count=cpcv.get("valid_trade_count", cpcv.get("n_trades", 0)),
        win_rate=cpcv.get("win_rate", 0.0),
        sharpe=cpcv.get("sharpe", 0.0),
        benchmark_roi_pct=cpcv.get("benchmark_roi_pct", 0.0),
        exposure_ratio=cpcv.get("exposure_ratio", 0.0)
    )
    
    passed, reason = validate_sample(metrics_obj)
    is_rejected = not passed
    rejection_reason = reason if is_rejected else None
    
    if not passed:
        hard_fail.append(reason)
        
    # [Evaluation Layer] 보조 검사 (선택 사항)
    cpcv_worst = cpcv.get("cpcv_worst", 0.0)
    if cpcv_worst < -2.0:
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
        is_rejected=is_rejected,
        rejection_reason=rejection_reason,
        verdict_dump=verdict_dump
    )

    return ledger_record, artifact


def run_experiment(
    registry: TemplateRegistry,
    policy_spec: PolicySpec,
    market_data: pd.DataFrame,
    ledger_repo: Optional[LedgerRepo] = None,
) -> tuple[LedgerRecord, ArtifactBundle]:
    
    if lgb is None:
        raise ImportError("LightGBM must be installed")
    
    import logging
    logging.getLogger("l2.ml_guard").setLevel(logging.WARNING)
    logging.getLogger("feature.registry").setLevel(logging.WARNING)
    logging.getLogger("feature.custom_loader").setLevel(logging.WARNING)
    
    if market_data.empty:
        raise ValueError("Market data is empty")
        
    df = market_data.copy()
    df.columns = [c.lower() for c in df.columns]

    t0 = time.time()
    
    X_features = _generate_features_cached(
        df_values=df.values,
        df_columns=df.columns.tolist(),
        df_index=df.index,
        genome=policy_spec.feature_genome
    )
    t1 = time.time()
    
    if X_features.empty:
        raise ValueError("No features generated from genome. Check definitions.")

    # Global Context Features Injection
    try:
        if len(df) > 200:
            sma_50 = df["close"].rolling(50).mean()
            sma_200 = df["close"].rolling(200).mean()
            
            trend_dir = pd.Series(0.0, index=df.index)
            trend_dir += (df["close"] > sma_50).astype(float) * 0.5
            trend_dir += (sma_50 > sma_200).astype(float) * 0.5
            trend_dir -= (df["close"] < sma_50).astype(float) * 0.5
            trend_dir -= (sma_50 < sma_200).astype(float) * 0.5
            
            X_features["_CTX_TrendScore"] = trend_dir.reindex(X_features.index).fillna(0.0)
        
        log_rets = np.log(df["close"] / df["close"].shift(1))
        vol_short = log_rets.rolling(20).std()
        vol_long = log_rets.rolling(100).std()
        vol_ratio = (vol_short / (vol_long + 1e-9)).clip(0, 5)
        X_features["_CTX_VolRatio"] = vol_ratio.reindex(X_features.index).fillna(1.0)
        
        if "volume" in df.columns:
            pv_corr = df["close"].rolling(20).corr(df["volume"]).fillna(0.0)
            X_features["_CTX_PVCorr"] = pv_corr.reindex(X_features.index).fillna(0.0)
        
        rolling_mean = df["close"].rolling(20).mean()
        rolling_std = df["close"].rolling(20).std()
        bb_lower = rolling_mean - (rolling_std * 2.5)
        shock_series = (df["close"] < bb_lower).astype(float)
        X_features["_CTX_ShockFlag"] = shock_series.reindex(X_features.index).fillna(0.0)
        
        logger.debug(f"[Experiment] Injected Optimized Global Context Features")
    except Exception as e:
        logger.warning(f"[Experiment] Context Injection failed: {e}")

    core = _run_experiment_core(
        policy_spec=policy_spec,
        df=df,
        X_features=X_features,
        risk_budget=policy_spec.risk_budget or {},
        include_trade_logic=True,
    )

    # [V9] Scorecard는 이제 순수 Backtest 결과
    scorecard = {
        "total_return_pct": core.get("total_return_pct", 0.0),
        "mdd_pct": core.get("mdd_pct", 0.0),
        "win_rate": core.get("win_rate", 0.0),
        "n_trades": core.get("n_trades", 0),
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

    total_time = time.time() - t0
    t_train = core.get("t_train", 0.0)
    logger.info(f"[Perf] Exp {policy_spec.archetype} | Total: {total_time:.2f}s | Feat: {t1-t0:.2f}s | Train: {t_train:.2f}s | Trades: {core.get('n_trades', 0)} | WinRate: {core.get('win_rate', 0):.1%}")

    return ledger_record, artifact
