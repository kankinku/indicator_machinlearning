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
from src.shared.logger import get_logger
from src.backtest.engine import DeterministicBacktestEngine
from src.orchestration.policy_evaluator import RuleEvaluator
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
    핵심 실험 실행 함수. (V13-PRO Rigorous Version)
    
    [V13-PRO]
    - Separate Entry/Exit logic from RuleEvaluator.
    - Deterministic State Machine in BacktestEngine.
    - Complexity-aware scoring.
    
    [V18]
    - FEATURE_MISSING 조기 감지 및 명시적 REJECT
    """
    # 1. Evaluate Signals from Rules
    evaluator = RuleEvaluator()
    entry_sig, exit_sig, complexity = evaluator.evaluate_signals(X_features, policy_spec)
    
    # [V18] FEATURE_MISSING 감지: complexity=-1.0은 LogicTree 매칭 실패 마커
    if complexity < 0:
        # 매칭 실패: 즉시 실패 결과 반환
        error_info = getattr(policy_spec, '_logictree_error', {})
        error_type = error_info.get("type", "unknown")
        error_key = error_info.get("feature_key", "unknown")
        
        logger.warning(f"[실험] 특징 누락 감지: {error_type} ('{error_key}')")
        
        # 실패 결과 구조 (REJECT 처리용)
        return {
            "results_df": pd.DataFrame(),
            "cpcv": {
                "n_trades": 0,
                "win_rate": 0.0,
                "total_return_pct": 0.0,
                "mdd_pct": 0.0,
                "profit_factor": 1.0,
                "exposure_ratio": 0.0,
                "avg_trade_return": 0.0,
                "complexity_score": 0.0,
                "trades_per_year": 0.0,
                "excess_return": 0.0,
                "oos_bars": len(df),
                # [V18] FEATURE_MISSING 명시적 마커
                "_feature_missing": True,
                "_feature_missing_type": error_type,
                "_feature_missing_key": error_key,
            },
            "pbo": 0.0,
            "turnover": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "total_return_pct": 0.0,
            "mdd_pct": 0.0,
            "trade_logic": {
                "error": f"FEATURE_MISSING: {error_type} for '{error_key}'",
                "complexity": 0.0,
            },
            "final_guard": None,
            "label_config": {},
            "label_hash": "v18_feature_missing",
            "t_train": 0.0,
            "bt_result": None,
            # [V18] 명시적 실패 마커
            "_rejected": True,
            "_rejection_reason": f"FEATURE_MISSING_{error_type.upper()}",
        }
    
    # 2. Run Deterministic Backtest
    from src.shared.backtest import derive_trade_params
    tp_pct, sl_pct, horizon = derive_trade_params(risk_budget)
    bt_engine = DeterministicBacktestEngine()
    bt_result = bt_engine.run(
        df["close"],
        entry_sig,
        exit_sig,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        max_hold_bars=horizon,
    )
    
    # 3. Format results for ledger
    results_df = pd.DataFrame({
        "entry_sig": entry_sig,
        "exit_sig": exit_sig,
        "pred": entry_sig, # For compatibility with V12/V14 evaluation orchestrator
        "pos": [0] * len(df) # Logic moved to engine, but kept for schema
    }, index=df.index)
    
    # Phase 1 Filter: Cheap Checks
    # [V13.5] Early rejection for extreme outliers
    if bt_result.trades_per_year < 2.0 or bt_result.trades_per_year > 300.0:
        pass # Rejection handled by validate_sample later
        
    from src.shared.observability import compute_cycle_stats
    _, cycle_stats = compute_cycle_stats(bt_result.trades, len(df))

    # metrics for scoring
    cpcv = {
        "n_trades": bt_result.trade_count,
        "cycle_count": cycle_stats.get("cycle_count", 0),
        "entry_count": cycle_stats.get("entry_count", 0),
        "exit_count": cycle_stats.get("exit_count", 0),
        "win_rate": bt_result.win_rate,
        "total_return_pct": bt_result.total_return,
        "mdd_pct": bt_result.mdd,
        "profit_factor": bt_result.profit_factor,
        "exposure_ratio": bt_result.exposure,
        "avg_trade_return": bt_result.avg_trade_return,
        "complexity_score": complexity,
        "trades_per_year": bt_result.trades_per_year, # [V13.5]
        "excess_return": bt_result.excess_return,     # [V13.5]
        "invalid_action_count": getattr(bt_result, "invalid_action_count", 0),
        "invalid_action_rate": getattr(bt_result, "invalid_action_rate", 0.0),
        "oos_bars": len(df),
    }

    # Trade Logic description
    trade_logic = {
        "rules": policy_spec.decision_rules,
        "complexity": complexity,
        "description": f"V13-PRO Deterministic Strategy | Rules: {policy_spec.decision_rules}"
    }
    
    return {
        "results_df": results_df,
        "cpcv": cpcv,
        "pbo": 0.0,
        "turnover": bt_result.trade_count / len(df) if len(df) > 0 else 0.0,
        "n_trades": bt_result.trade_count,
        "win_rate": bt_result.win_rate,
        "total_return_pct": bt_result.total_return,
        "mdd_pct": bt_result.mdd,
        "trade_logic": trade_logic,
        "final_guard": None,
        "label_config": {},
        "label_hash": "v13_pro_deterministic",
        "t_train": 0.0,
        "bt_result": bt_result,
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

    # =========================================================
    # [V18] FEATURE_MISSING 조기 감지 및 즉시 REJECT
    # =========================================================
    if core.get("_rejected") or cpcv.get("_feature_missing"):
        rejection_reason = core.get("_rejection_reason") or "FEATURE_MISSING"
        exp_id = str(uuid4())
        
        artifact = ArtifactBundle(
            label_config=label_config,
            direction_model=None,
            risk_model=None,
            calibration_metrics={},
            metadata={"genome": policy_spec.feature_genome, "error": rejection_reason},
            backtest_results=[]
        )
        
        ledger_record = LedgerRecord(
            exp_id=exp_id,
            timestamp=time.time(),
            policy_spec=policy_spec,
            data_hash=_hash_payload(df.values.tobytes()) if not df.empty else "empty",
            feature_hash="",
            label_hash=label_hash,
            model_artifact_ref="",
            cpcv_metrics=cpcv,
            pbo=pbo,
            risk_report={"ok": False},
            reason_codes=[rejection_reason],
            is_rejected=True,
            rejection_reason=rejection_reason,
            verdict_dump=verdict_dump
        )
        
        return ledger_record, artifact

    # [V9] Validation based on Backtest results
    # [vNext] Validation Layer Integration (Hard Gates)
    hard_fail = []
    
    # [V16] Convert dict to SampleMetrics for validation
    from src.shared.metrics import TradeStats, EquityStats
    
    # [V18] Defensive extraction: handle both 'sample_' prefix and standard keys
    n_trades = cpcv.get("sample_trades") if "sample_trades" in cpcv else cpcv.get("n_trades", 0)
    win_rate = cpcv.get("sample_win_rate") if "sample_win_rate" in cpcv else cpcv.get("win_rate", 0.0)
    total_ret = cpcv.get("sample_total_return_pct") if "sample_total_return_pct" in cpcv else cpcv.get("total_return_pct", 0.0)
    mdd = cpcv.get("sample_mdd_pct") if "sample_mdd_pct" in cpcv else cpcv.get("mdd_pct", 0.0)
    sharpe = cpcv.get("sample_sharpe") if "sample_sharpe" in cpcv else cpcv.get("sharpe", 0.0)
    rr = cpcv.get("sample_rr") if "sample_rr" in cpcv else cpcv.get("reward_risk", 1.0)
    benchmark_roi = cpcv.get("sample_benchmark_roi_pct") if "sample_benchmark_roi_pct" in cpcv else cpcv.get("benchmark_roi_pct", 0.0)

    metrics_obj = SampleMetrics(
        window_id="BACKTEST",
        trades=TradeStats(
            trade_count=n_trades,
            valid_trade_count=cpcv.get("valid_trade_count", n_trades),
            win_rate=win_rate,
            reward_risk=rr,
            cycle_count=int(cpcv.get("cycle_count", n_trades)),
            entry_count=int(cpcv.get("entry_count", n_trades)),
            exit_count=int(cpcv.get("exit_count", n_trades)),
            invalid_action_count=int(cpcv.get("invalid_action_count", 0)),
            invalid_action_rate=float(cpcv.get("invalid_action_rate", 0.0)),
        ),
        equity=EquityStats(
            total_return_pct=total_ret,
            max_drawdown_pct=mdd,
            sharpe=sharpe,
            benchmark_roi_pct=benchmark_roi,
            exposure_ratio=cpcv.get("exposure_ratio", 0.0)
        ),
        raw_score=cpcv.get("eval_score", 0.0),
        bars_total=cpcv.get("oos_bars", len(df))
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
        
        logger.debug("[실험] 최적화된 글로벌 컨텍스트 주입")
    except Exception as e:
        logger.warning(f"[실험] 컨텍스트 주입 실패: {e}")

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
    logger.info(
        f"[성능] {policy_spec.archetype} | 총 {total_time:.2f}s | 특징 {t1-t0:.2f}s | 학습 {t_train:.2f}s "
        f"| 거래 {core.get('n_trades', 0)} | 승률 {core.get('win_rate', 0):.1%}"
    )

    return ledger_record, artifact
