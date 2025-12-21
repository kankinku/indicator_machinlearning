import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
from dataclasses import asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import Config and Registry
from src.config import config
from src.ledger.repo import LedgerRepo
from src.features.registry import FeatureRegistry
from src.shared.returns import (
    compute_compounded_return_pct,
    get_risk_reward_ratio,
    get_risk_unit,
)
from src.shared.backtest import run_signal_backtest
from src.shared.ranking import check_return_stability

app = FastAPI(title="Vibe Trading Lab API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services
LEDGER_PATH = PROJECT_ROOT / "ledger"
repo = LedgerRepo(LEDGER_PATH)

feature_registry = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH))
# Ensure registry is initialized (loaded)
feature_registry.initialize()

# Feature Factory & Data Loading
from src.features.factory import FeatureFactory
from src.data.loader import DataLoader

feature_factory = FeatureFactory()
MARKET_DATA_CACHE = None
FULL_MARKET_DATA_CACHE = None

def get_cached_market_data():
    global MARKET_DATA_CACHE
    if MARKET_DATA_CACHE is None:
        print("Loading Market Data for Preview...")
        loader = DataLoader()
        # Fetch last 500 bars for performance
        # We can't limit fetch_all easily by rows, so we slice after
        df = loader.fetch_all()
        MARKET_DATA_CACHE = df.iloc[-500:] # Keep last 500 for visualization
    return MARKET_DATA_CACHE

def get_full_market_data():
    global FULL_MARKET_DATA_CACHE
    if FULL_MARKET_DATA_CACHE is None:
        loader = DataLoader()
        FULL_MARKET_DATA_CACHE = loader.fetch_all()
    return FULL_MARKET_DATA_CACHE

CODE_MAP = {
    "DD_LIMIT_BREACH": "Max Drawdown Exceeded",
    "CPCV_WORST_TOO_LOW": "Validation Return Low",
    "LABEL_IMBALANCE": "Data Imbalance",
    "TURNOVER_TOO_HIGH": "Excessive Turnover",
    "SHARPE_TOO_LOW": "Low Sharpe Ratio",
    "PBO_TOO_HIGH": "Overfitting Risk"
}

def format_strategy_name(template_id, genome):
    if "EVO" in template_id:
        return "Evolutionary", "Evo"
    elif "RANDOM" in template_id:
        return "Random Search", "Rnd"
    elif "GENOME" in template_id:
        return "Manual/Legacy", "Leg"
    else:
        return template_id, "AI"

def get_genome_desc(genome):
    if not isinstance(genome, dict):
        return str(genome)
    
    items = []
    keys = sorted(list(genome.keys()))
    
    for k in keys:
        params = genome[k]
        p_str = ""
        if isinstance(params, dict):
            # Try to find 'window' or 'length'
            main_val = params.get('window') or params.get('length') or params.get('fast')
            if main_val:
                p_str = f"({main_val})"
        items.append(f"{k}{p_str}")
        
    if len(items) > 3:
        return f"{items[0]}, {items[1]} +{len(items)-2}"
    return ", ".join(items)

@app.get("/api/experiments")
def get_experiments():
    try:
        records_obj = repo.load_records()
    except Exception as e:
        print(f"Error loading records: {e}")
        return []

    if not records_obj:
        return []

    flat_data = []
    market_df = None
    for r in records_obj:
        # Meta Data
        origin_full, origin_tag = format_strategy_name(r.policy_spec.template_id, r.policy_spec.feature_genome)
        genome_txt = get_genome_desc(r.policy_spec.feature_genome)
        risk_budget = r.policy_spec.risk_budget or {}
        
        # Metrics
        metrics = r.cpcv_metrics if r.cpcv_metrics else {}
        ret_mean = metrics.get("cpcv_mean", 0.0)
        vol_std = metrics.get("cpcv_std", 0.0)
        cpcv_worst = metrics.get("cpcv_worst", 0.0)
        n_trades = metrics.get("n_trades", 0)
        win_rate = metrics.get("win_rate", 0.0)
        sharpe = ret_mean / (vol_std + 1e-9) if vol_std > 0 else 0.0
        stability_pass, vol_ratio = check_return_stability(
            cpcv_mean=ret_mean,
            cpcv_std=vol_std,
            cpcv_worst=cpcv_worst,
            win_rate=win_rate,
            n_trades=n_trades,
        )
        
        # Reasons
        status = "Approved" if not r.reason_codes else "Rejected"
        fail_reason = r.reason_codes[0] if r.reason_codes else ""
        fail_reason_clean = CODE_MAP.get(fail_reason, fail_reason)
        
        # Risk Params
        risk_unit = get_risk_unit(risk_budget)
        risk_reward_ratio = get_risk_reward_ratio(risk_budget)
        stop_loss_pct = risk_unit * 100.0 if risk_unit is not None else None
        target_return_pct = None
        if risk_unit is not None and risk_reward_ratio is not None:
            target_return_pct = risk_unit * risk_reward_ratio * 100.0

        # Backtest-derived metrics (preferred)
        bt_total_return = metrics.get("sample_total_return_pct")
        bt_win_rate = metrics.get("sample_win_rate")
        bt_trades = metrics.get("sample_trades")
        bt_mdd = metrics.get("sample_mdd_pct")

        if bt_total_return is None:
            results_path = LEDGER_PATH / "artifacts" / f"{r.exp_id}_results.csv"
            if results_path.exists():
                if market_df is None:
                    market_df = get_full_market_data()
                if not market_df.empty:
                    results_df = pd.read_csv(results_path)
                    bt = run_signal_backtest(
                        price_df=market_df,
                        results_df=results_df,
                        risk_budget=risk_budget,
                        cost_bps=r.policy_spec.execution_assumption.get("cost_bps", 5),
                    )
                    bt_total_return = bt.total_return_pct
                    bt_win_rate = bt.win_rate
                    bt_trades = bt.trade_count
                    bt_mdd = bt.mdd_pct

        if bt_total_return is None:
            bt_total_return = compute_compounded_return_pct(
                exp_id=r.exp_id,
                model_artifact_ref=r.model_artifact_ref,
                ledger_dir=LEDGER_PATH,
                risk_budget=risk_budget,
            )
        if bt_total_return is None:
            bt_total_return = ret_mean * n_trades * (risk_unit * 100.0)

        if bt_win_rate is None:
            bt_win_rate = win_rate
        if bt_trades is None:
            bt_trades = n_trades
        if bt_mdd is None:
            bt_mdd = 0.0

        # Avg Return (%)
        return_mean = (bt_total_return / bt_trades) if bt_trades else 0.0

        ts_iso = pd.to_datetime(r.timestamp, unit='s').isoformat()

        flat_data.append({
            "id": r.exp_id,
            "short_id": r.exp_id[:8],
            "timestamp": ts_iso,
            "origin": origin_full,
            "origin_tag": origin_tag,
            "indicators": genome_txt,
            "genome_full": r.policy_spec.feature_genome,
            "template_id": r.policy_spec.template_id,
            "data_window": r.policy_spec.data_window,
            "execution_assumption": r.policy_spec.execution_assumption,
            "risk_budget": risk_budget,
            "rl_meta": r.policy_spec.rl_meta,
            "sharpe": float(sharpe),
            "return_mean": float(return_mean),
            "total_return": float(bt_total_return),
            "volatility": float(vol_std),
            "cpcv_mean": float(ret_mean),
            "cpcv_std": float(vol_std),
            "cpcv_worst": float(cpcv_worst),
            "vol_ratio": float(vol_ratio),
            "stability_pass": bool(stability_pass),
            "trades": int(bt_trades),
            "win_rate": float(bt_win_rate),
            "status": status,
            "fail_reason": fail_reason_clean,
            "module_key": metrics.get("module_key"),
            "eval_stage": metrics.get("eval_stage"),
            "eval_score": metrics.get("eval_score"),
            "sample_id": metrics.get("sample_id"),
            "sample_window_id": metrics.get("sample_window_id"),
            "sample_total_return_pct": metrics.get("sample_total_return_pct"),
            "sample_mdd_pct": metrics.get("sample_mdd_pct"),
            "sample_rr": metrics.get("sample_rr"),
            "sample_vol_pct": metrics.get("sample_vol_pct"),
            "sample_trades": metrics.get("sample_trades"),
            "sample_win_rate": metrics.get("sample_win_rate"),
            "sample_mdd_pct": metrics.get("sample_mdd_pct"),
            "backtest_mdd_pct": float(bt_mdd),
            "k_up": risk_budget.get("k_up"),
            "k_down": risk_budget.get("k_down"),
            "horizon": risk_budget.get("horizon"),
            "risk_profile": risk_budget.get("risk_profile"),
            "tp_pct": risk_budget.get("tp_pct"),
            "sl_pct": risk_budget.get("sl_pct"),
            "stop_loss_pct": stop_loss_pct,
            "target_return_pct": target_return_pct,
            "risk_reward_ratio": risk_reward_ratio,
            "entry_threshold": r.policy_spec.tuned_params.get("entry_threshold", config.EVAL_ENTRY_THRESHOLD),
            "entry_max_prob": config.EVAL_ENTRY_MAX_PROB,
        })
        
    # Sort for generation assignment
    flat_data.sort(key=lambda x: x["timestamp"]) 
    for i, item in enumerate(flat_data):
        item["generation"] = i + 1
        
    flat_data.sort(key=lambda x: x["timestamp"], reverse=True) 

    return flat_data

@app.get("/api/experiments/{exp_id}/chart")
def get_chart_data(exp_id: str):
    csv_path = LEDGER_PATH / "artifacts" / f"{exp_id}_results.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Chart data not found")
    
    try:
        results_df = pd.read_csv(csv_path)
        record = next((r for r in repo.load_records() if r.exp_id == exp_id), None)
        risk_budget = record.policy_spec.risk_budget if record else {}
        cost_bps = record.policy_spec.execution_assumption.get("cost_bps", 5) if record else 5

        market_df = get_full_market_data()
        if market_df.empty:
            raise HTTPException(status_code=500, detail="Market data unavailable")

        bt = run_signal_backtest(
            price_df=market_df,
            results_df=results_df,
            risk_budget=risk_budget,
            cost_bps=cost_bps,
        )

        return {
            "dates": bt.dates,
            "equity": bt.equity_pct
        }
    except Exception as e:
        print(f"Error reading csv: {e}")
        raise HTTPException(status_code=500, detail="Error reading chart data")

@app.get("/api/experiments/{exp_id}/backtest")
def get_backtest_result(exp_id: str):
    csv_path = LEDGER_PATH / "artifacts" / f"{exp_id}_results.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Backtest data not found")

    try:
        results_df = pd.read_csv(csv_path)
        record = next((r for r in repo.load_records() if r.exp_id == exp_id), None)
        if record is None:
            raise HTTPException(status_code=404, detail="Experiment not found")

        risk_budget = record.policy_spec.risk_budget or {}
        cost_bps = record.policy_spec.execution_assumption.get("cost_bps", 5)
        market_df = get_full_market_data()
        if market_df.empty:
            raise HTTPException(status_code=500, detail="Market data unavailable")

        bt = run_signal_backtest(
            price_df=market_df,
            results_df=results_df,
            risk_budget=risk_budget,
            cost_bps=cost_bps,
        )
        entry_signals = int((results_df.get("pred", pd.Series(dtype=int)) != 0).sum())
        start_date = bt.dates[0] if bt.dates else "-"
        end_date = bt.dates[-1] if bt.dates else "-"

        return {
            "success": True,
            "metrics": {
                "total_return_pct": float(bt.total_return_pct),
                "mdd_pct": float(bt.mdd_pct),
                "win_rate": float(bt.win_rate),
                "trade_count": int(bt.trade_count),
                "entry_signals": entry_signals,
                "start_date": start_date,
                "end_date": end_date,
            },
            "trades": bt.trades[-20:],
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail="Backtest failed")

@app.get("/api/features")
def get_features():
    """Returns the list of all registered features (indicators)."""
    try:
        features = feature_registry.list_all()
        # Convert Dataclasses to dicts
        return [asdict(f) for f in features]
    except Exception as e:
        print(f"Error listing features: {e}")
        return []

@app.post("/api/features/{feature_id}/preview")
def preview_feature(feature_id: str, params: Optional[dict] = None):
    """
    Calculates the indicator on the last 500 bars of market data.
    Returns: { dates: [], values: { "col1": [], "col2": [] } }
    """
    try:
        # 1. Get Data
        df = get_cached_market_data()
        if df.empty:
             raise HTTPException(status_code=500, detail="Market data unavailble")
        
        # 2. Prepare Default Params if needed
        if not params:
            # Fetch defaults from registry
            meta = feature_registry.get(feature_id)
            if not meta:
                raise HTTPException(status_code=404, detail="Feature not found")
            params = {p.name: p.default for p in meta.params}

        # 3. Calculate Feature
        # Construct a mini-genome for this single feature
        genome = {feature_id: params}
        
        # Generate
        result_df = feature_factory.generate_from_genome(df, genome)
        
        if result_df.empty:
             raise HTTPException(status_code=500, detail="Feature calculation returned empty")
             
        # 4. Format for Chart
        dates = df.index.strftime('%Y-%m-%d').tolist()
        
        # The result_df columns are like "FEATURE_ID__col"
        # We want to separate them for plotting
        values = {}
        for col in result_df.columns:
            # Remove the feature_id prefix for cleaner display if desired, 
            # or keep it. Let's make it cleaner: "RSI_A1B2__rsi" -> "rsi"
            clean_name = col.split("__")[-1] if "__" in col else col
            values[clean_name] = result_df[col].tolist()
            
        # Also return Close price for context
        close_price = df['close'].tolist() if 'close' in df.columns else []

        return {
            "dates": dates,
            "values": values,
            "close": close_price
        }

    except Exception as e:
        print(f"Preview Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel

class BacktestRequest(BaseModel):
    feature_ids: List[str]

@app.post("/api/backtest")
def run_backtest(request: BacktestRequest):
    """
    Runs a simplified backtest using selected features.
    Uses existing experiment runner logic but returns results directly.
    """
    from src.contracts import PolicySpec
    from uuid import uuid4
    from src.templates.registry import TemplateRegistry
    from src.orchestration.run_experiment import run_experiment
    
    try:
        # 1. Get Market Data (Full for proper training)
        loader = DataLoader()
        df = loader.fetch_all()
        
        if df.empty:
            raise HTTPException(status_code=500, detail="Market data unavailable")
        
        # 2. Build Genome from selected features
        genome = {}
        for fid in request.feature_ids:
            meta = feature_registry.get(fid)
            if meta:
                # Use default params
                params = {p.name: p.default for p in meta.params}
                genome[fid] = params
        
        if not genome:
            raise HTTPException(status_code=400, detail="No valid features selected")
        
        # 3. Create PolicySpec
        policy = PolicySpec(
            spec_id=str(uuid4()),
            template_id="WEB_BACKTEST",
            feature_genome=genome,
            risk_budget={
                "k_up": 1.0,
                "k_down": 1.0,
                "horizon": 20,
                "risk_profile": "DEFAULT",
                "stop_loss": 0.015,
                "risk_reward_ratio": 1.0,
            },
            execution_assumption={"cost_bps": 5}
        )
        
        # 4. Run Experiment
        registry = TemplateRegistry()  # Stub, not used in new logic
        record, artifact = run_experiment(registry, policy, df)
        
        # 5. Extract Results
        metrics = record.cpcv_metrics or {}
        ret_mean = metrics.get("cpcv_mean", 0.0)
        vol_std = metrics.get("cpcv_std", 0.0)
        sharpe = ret_mean / (vol_std + 1e-9) if vol_std > 0 else 0.0
        
        # Backtest using entry/exit/stop rules
        results_df = pd.DataFrame(artifact.backtest_results or [])
        bt = run_signal_backtest(
            price_df=df,
            results_df=results_df,
            risk_budget=policy.risk_budget,
            cost_bps=policy.execution_assumption.get("cost_bps", 5),
        )

        return {
            "success": True,
            "metrics": {
                "sharpe": float(sharpe),
                "win_rate": float(bt.win_rate),
                "n_trades": int(bt.trade_count),
                "total_return": float(bt.total_return_pct),
                "mdd_pct": float(bt.mdd_pct),
                "pbo": float(record.pbo)
            },
            "equity": bt.equity_pct,
            "dates": bt.dates,
            "trades": bt.trades[-20:],
            "reason_codes": record.reason_codes
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/experiments/{exp_id}/test")
def test_model(exp_id: str):
    """
    Runs a live prediction using a saved model.
    Loads the artifact, fetches latest market data, and generates a signal.
    """
    from src.l2_sl.artifacts import ArtifactBundle
    import joblib
    
    try:
        # 1. Find the record
        records = repo.load_records()
        record = next((r for r in records if r.exp_id == exp_id), None)
        
        if not record:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        # 2. Load the artifact (model)
        artifact_path = LEDGER_PATH / "artifacts" / f"{exp_id}.json"
        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail="Model artifact not found")
        
        artifact = ArtifactBundle.load(artifact_path)
        model = artifact.direction_model
        
        if model is None or isinstance(model, dict):
            raise HTTPException(status_code=400, detail="Model not available for testing")
        
        # 3. Get latest market data
        loader = DataLoader()
        df = loader.fetch_all()
        
        if df.empty:
            raise HTTPException(status_code=500, detail="Market data unavailable")
        
        # 4. Generate features using the same genome
        genome = record.policy_spec.feature_genome
        result_df = feature_factory.generate_from_genome(df, genome)
        
        if result_df.empty:
            raise HTTPException(status_code=500, detail="Feature generation failed")
        
        # 5. Get latest row for prediction
        latest_row = result_df.iloc[[-1]]
        latest_date = str(df.index[-1].date()) if hasattr(df.index[-1], 'date') else str(df.index[-1])
        
        # 6. Predict
        try:
            # MLGuard wrapper
            if hasattr(model, 'predict'):
                pred_result = model.predict(latest_row, threshold=0.6, max_prob=0.9)
                signal = int(pred_result['signal'].iloc[0])
                scale = float(pred_result['scale'].iloc[0])
                confidence = float(pred_result.get('confidence', pd.Series([0.7])).iloc[0]) if 'confidence' in pred_result else 0.7
            else:
                # Direct LightGBM model
                probs = model.predict(latest_row)
                if len(probs.shape) > 1:
                    pred_class = int(probs[0].argmax())
                    confidence = float(probs[0].max())
                else:
                    pred_class = int(probs[0])
                    confidence = 0.7
                # Map back from 0,1,2 to -1,0,1
                signal = pred_class - 1
                scale = 1.0 if confidence > 0.7 else 0.5
        except Exception as pred_err:
            print(f"Prediction error: {pred_err}")
            signal = 0
            scale = 0.0
            confidence = 0.0
        
        return {
            "signal": signal,
            "scale": scale,
            "confidence": confidence,
            "latest_date": latest_date,
            "exp_id": exp_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Serve static files (Frontend)
WEB_ROOT = PROJECT_ROOT / "web"
if not WEB_ROOT.exists():
    WEB_ROOT.mkdir()

app.mount("/", StaticFiles(directory=str(WEB_ROOT), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
