
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Ensure src is importable
sys.path.append(str(Path(__file__).parent))
from src.ledger.repo import LedgerRepo
from src.shared.returns import get_risk_reward_ratio, get_risk_unit
from src.shared.backtest import run_signal_backtest
from src.data.loader import DataLoader
from src.shared.ranking import check_return_stability

# --- Constants & Registry ---
CODE_MAP = {
    "DD_LIMIT_BREACH": "Max Drawdown Exceeded",
    "CPCV_WORST_TOO_LOW": "Validation Return Low",
    "LABEL_IMBALANCE": "Data Imbalance",
    "TURNOVER_TOO_HIGH": "Excessive Turnover",
    "SHARPE_TOO_LOW": "Low Sharpe Ratio",
    "PBO_TOO_HIGH": "Overfitting Risk"
}

# Initialize Repo
LEDGER_PATH = Path("./ledger").resolve()
repo = LedgerRepo(LEDGER_PATH)

@st.cache_data(ttl=300)
def load_full_market_data():
    loader = DataLoader()
    return loader.fetch_all()

def format_strategy_name(template_id, genome):
    """Clean formatting for strategy display without emojis"""
    if "EVO" in template_id:
        return "Evolutionary", "Evo"
    elif "RANDOM" in template_id:
        return "Random Search", "Rnd"
    elif "GENOME" in template_id:
        return "Manual/Legacy", "Leg"
    else:
        # RL Action Name
        return template_id, "AI"

def get_genome_desc(genome):
    """
    Format genome into a readable summary string.
    Example: {RSI: {window:14}} -> "RSI(14)"
    """
    if not isinstance(genome, dict):
        return str(genome)
        
    items = []
    keys = sorted(list(genome.keys()))
    
    for k in keys:
        params = genome[k]
        # Extract main param if possible
        # Heuristic: First param value or 'window'
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

@st.cache_data(ttl=2)
def load_data():
    try:
        records_obj = repo.load_records()
    except Exception:
        return pd.DataFrame()

    if not records_obj:
        return pd.DataFrame()

    flat_data = []
    market_df = None
    for r in records_obj:
        # Meta Data
        origin_full, origin_tag = format_strategy_name(r.policy_spec.template_id, r.policy_spec.feature_genome)
        genome_txt = get_genome_desc(r.policy_spec.feature_genome)
        risk_budget = r.policy_spec.risk_budget or {}
        
        # Metrics
        ret_mean = r.cpcv_metrics.get("cpcv_mean", 0.0) if r.cpcv_metrics else 0.0
        vol_std = r.cpcv_metrics.get("cpcv_std", 0.0) if r.cpcv_metrics else 0.0
        cpcv_worst = r.cpcv_metrics.get("cpcv_worst", 0.0) if r.cpcv_metrics else 0.0
        n_trades = r.cpcv_metrics.get("n_trades", 0) if r.cpcv_metrics else 0
        win_rate = r.cpcv_metrics.get("win_rate", 0.0) if r.cpcv_metrics else 0.0
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
        
        # Risk Params (for display and return conversion)
        risk_unit = get_risk_unit(risk_budget)
        risk_reward_ratio = get_risk_reward_ratio(risk_budget)
        stop_loss_pct = risk_unit * 100.0 if risk_unit is not None else None
        target_return_pct = None
        if risk_unit is not None and risk_reward_ratio is not None:
            target_return_pct = risk_unit * risk_reward_ratio * 100.0

        # Backtest-derived metrics (preferred)
        metrics = r.cpcv_metrics or {}
        bt_total_return = metrics.get("sample_total_return_pct")
        bt_win_rate = metrics.get("sample_win_rate")
        bt_trades = metrics.get("sample_trades")
        bt_mdd = metrics.get("sample_mdd_pct")

        if bt_total_return is None:
            results_path = LEDGER_PATH / "artifacts" / f"{r.exp_id}_results.csv"
            if results_path.exists():
                if market_df is None:
                    market_df = load_full_market_data()
                if market_df is not None and not market_df.empty:
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
            bt_total_return = 0.0
        if bt_win_rate is None:
            bt_win_rate = 0.0
        if bt_trades is None:
            bt_trades = 0
        if bt_mdd is None:
            bt_mdd = 0.0

        # Avg Return (%)
        return_mean = (bt_total_return / bt_trades) if bt_trades else 0.0

        flat_data.append({
            "id": r.exp_id,
            "short_id": r.exp_id[:8],
            "timestamp": pd.to_datetime(r.timestamp, unit='s'),
            "origin": origin_full,
            "origin_tag": origin_tag,
            "indicators": genome_txt,
            "genome_full": r.policy_spec.feature_genome,
            "sharpe": sharpe,
            "return_mean": return_mean,
            "total_return": bt_total_return,
            "volatility": vol_std,
            "cpcv_mean": ret_mean,
            "cpcv_std": vol_std,
            "cpcv_worst": cpcv_worst,
            "vol_ratio": vol_ratio,
            "stability_pass": stability_pass,
            "trades": bt_trades,
            "win_rate": bt_win_rate,
            "status": status,
            "fail_reason": fail_reason_clean,
            "verdict": r.verdict_dump,
            "backtest_mdd_pct": bt_mdd,
            "k_up": risk_budget.get("k_up"),
            "k_down": risk_budget.get("k_down"),
            "horizon": risk_budget.get("horizon"),
            "risk_profile": risk_budget.get("risk_profile"),
            "stop_loss_pct": stop_loss_pct,
            "target_return_pct": target_return_pct,
            "risk_reward_ratio": risk_reward_ratio,
            # Holistic Score: Return + (WinRate*50) + (Trades*0.1)
            "holistic_score": (bt_total_return or 0.0) + ((bt_win_rate or 0.0) * 50.0) + ((bt_trades or 0) * 0.1),
            # For Search
            "search_label": f"[{origin_tag}] Sharpe:{sharpe:.2f} | {genome_txt} ({r.exp_id[:8]})"
        })
        
    df = pd.DataFrame(flat_data)
    if not df.empty:
        # Calculate Generation (1 = Oldest, N = Newest)
        df = df.sort_values("timestamp", ascending=True)
        df["generation"] = range(1, len(df) + 1)
        
        # Final Sort for Display (Newest First)
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df

@st.cache_data(ttl=2)
def load_chart_data(exp_id):
    csv_path = LEDGER_PATH / "artifacts" / f"{exp_id}_results.csv"
    if csv_path.exists():
        results_df = pd.read_csv(csv_path)
        record = next((r for r in repo.load_records() if r.exp_id == exp_id), None)
        risk_budget = record.policy_spec.risk_budget if record else {}
        cost_bps = record.policy_spec.execution_assumption.get("cost_bps", 5) if record else 5
        market_df = load_full_market_data()
        if market_df is None or market_df.empty:
            return pd.DataFrame()
        bt = run_signal_backtest(
            price_df=market_df,
            results_df=results_df,
            risk_budget=risk_budget,
            cost_bps=cost_bps,
        )
        if not bt.dates:
            return pd.DataFrame()
        idx = pd.to_datetime(bt.dates)
        return pd.DataFrame({"equity": bt.equity_pct}, index=idx)
    return pd.DataFrame()

def render_sidebar():
    pass

def load_css():
    st.markdown("""
<style>
    @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css");

    /* Global Reset & Dark Mode Base */
    .stApp {
        background-color: #0F1113;
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
        color: #E5E8EB;
    }
    
    /* Utility: Monospace for Numbers */
    .mono {
        font-family: "SF Mono", "Roboto Mono", Menlo, Consolas, monospace !important;
        letter-spacing: -0.5px;
    }

    /* Keep Sidebar Visible but styled */
    [data-testid="stSidebar"] {
        background-color: #191F28;
        border-right: 1px solid #282F3A;
    }
    
    /* Hide Default Header/Toolbar */
    [data-testid="stHeader"], 
    [data-testid="stToolbar"], 
    [data-testid="stAppToolbar"],
    .stAppToolbar,
    header, 
    [data-testid="stDecoration"],
    .stDecoration {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
    
    /* Text Colors */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #E5E8EB;
    }
    p.small-text {
        color: #8B95A1 !important;
        font-size: 14px;
    }

    /* Layout */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1300px;
    }

    /* Premium Cards (Toss Style) */
    div.toss-card {
        background-color: #191F28;
        border-radius: 20px;
        padding: 24px;
        border: 1px solid #282F3A;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    /* Tables Readability Fix */
    div[data-testid="stDataFrame"] {
        background-color: #191F28 !important;
        border-radius: 12px;
        border: 1px solid #333D4B !important;
    }
    div[data-testid="stTable"] {
        border-radius: 12px;
        overflow: hidden;
    }
    table {
        color: #FFFFFF !important;
    }
    thead tr th {
        background-color: #202632 !important;
        color: #8B95A1 !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #333D4B !important;
    }
    tbody tr td {
        background-color: #191F28 !important;
        color: #FFFFFF !important;
        border-bottom: 1px solid #282F3A !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #191F28;
        border-radius: 16px;
        padding: 16px;
        border: 1px solid #282F3A;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        min-height: 115px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    label[data-testid="stMetricLabel"] {
        font-size: 13px;
        color: #8B95A1 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 700;
        color: #FFFFFF !important;
        font-family: inherit !important;
    }
    
    /* Buttons */
    div.stButton > button {
        background-color: #3182F6;
        color: white !important;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.5rem 1.0rem;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #1B64DA;
    }
    
    /* Status Badges */
    .badge-blue {
        background-color: rgba(49, 130, 246, 0.15);
        color: #3182F6;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
    }
    .badge-red {
        background-color: rgba(240, 68, 82, 0.15);
        color: #F04452;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
    }

    /* Custom Genome Scroll Container */
    .genome-scroll-container {
        max-height: 500px;
        overflow-y: auto;
        padding-right: 12px;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .genome-scroll-container::-webkit-scrollbar {
        width: 6px;
    }
    .genome-scroll-container::-webkit-scrollbar-track {
        background: transparent;
    }
    .genome-scroll-container::-webkit-scrollbar-thumb {
        background: #333D4B;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

