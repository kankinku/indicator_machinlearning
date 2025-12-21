
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Ensure src is importable
sys.path.append(str(Path(__file__).parent))
from src.ledger.repo import LedgerRepo

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
    for r in records_obj:
        # Meta Data
        origin_full, origin_tag = format_strategy_name(r.policy_spec.template_id, r.policy_spec.feature_genome)
        genome_txt = get_genome_desc(r.policy_spec.feature_genome)
        
        # Metrics
        ret_mean = r.cpcv_metrics.get("cpcv_mean", 0.0) if r.cpcv_metrics else 0.0
        vol_std = r.cpcv_metrics.get("cpcv_std", 0.0) if r.cpcv_metrics else 0.0
        n_trades = r.cpcv_metrics.get("n_trades", 0) if r.cpcv_metrics else 0
        win_rate = r.cpcv_metrics.get("win_rate", 0.0) if r.cpcv_metrics else 0.0
        sharpe = ret_mean / (vol_std + 1e-9) if vol_std > 0 else 0.0
        
        # Reasons
        status = "Approved" if not r.reason_codes else "Rejected"
        fail_reason = r.reason_codes[0] if r.reason_codes else ""
        fail_reason_clean = CODE_MAP.get(fail_reason, fail_reason)
        
        # Total Return (Cumulative)
        total_return = ret_mean * n_trades

        flat_data.append({
            "id": r.exp_id,
            "short_id": r.exp_id[:8],
            "timestamp": pd.to_datetime(r.timestamp, unit='s'),
            "origin": origin_full,
            "origin_tag": origin_tag,
            "indicators": genome_txt,
            "genome_full": r.policy_spec.feature_genome,
            "sharpe": sharpe,
            "return_mean": ret_mean,   # Renamed for clarity
            "total_return": total_return, # New Metric
            "volatility": vol_std,
            "trades": n_trades,
            "win_rate": win_rate,
            "status": status,
            "fail_reason": fail_reason_clean,
            "verdict": r.verdict_dump,
            # For Search
            "search_label": f"[{origin_tag}] Sharpe:{sharpe:.2f} | {genome_txt} ({r.exp_id[:8]})"
        })
        
    df = pd.DataFrame(flat_data)
    if not df.empty:
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df

@st.cache_data(ttl=2)
def load_chart_data(exp_id):
    csv_path = LEDGER_PATH / "artifacts" / f"{exp_id}_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    return pd.DataFrame()

def load_css():
    st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    /* Global Reset & Dark Mode Base */
    .stApp {
        background-color: #000000;
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
        color: #FFFFFF;
    }
    
    /* Utility: Monospace for Numbers (User Request) */
    .mono {
        font-family: "SF Mono", "Roboto Mono", Menlo, Consolas, monospace !important;
        letter-spacing: -0.5px;
    }
    
    /* Hide Streamlit Default UI Elements (Toolbar, Footer, Decoration) */
    [data-testid="stHeader"], 
    [data-testid="stToolbar"], 
    .stAppToolbar, 
    [data-testid="stFooter"], 
    header, footer {
        display: none !important;
        visibility: hidden !important;
    }
    .stApp > header {
        display: none !important;
    }
    
    /* Text Colors */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #FFFFFF;
    }
    p.small-text {
        color: #8B95A1 !important;
        font-size: 14px;
    }

    /* Layout & Spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1400px;
    }

    /* Typography */
    h1 {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF 0%, #8B95A1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    h2 {
        font-size: 24px;
        font-weight: 700;
        color: #F2F4F6 !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    h3 {
        font-size: 18px;
        font-weight: 600;
        color: #E5E8EB !important;
    }

    /* Premium Cards (Toss Style) */
    div.toss-card {
        background-color: #191F28;
        border-radius: 20px;
        padding: 24px;
        border: 1px solid #333D4B;
        margin-bottom: 20px;
        transition: all 0.2s ease-in-out;
    }
    div.toss-card:hover {
        border-color: #4E5968;
        background-color: #202632;
        transform: translateY(-2px);
    }

    /* Metrics styling */
    div[data-testid="stMetric"] {
        background-color: #191F28;
        border-radius: 16px;
        padding: 16px;
        border: 1px solid #333D4B;
    }
    label[data-testid="stMetricLabel"] {
        font-size: 13px;
        color: #8B95A1 !important;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 700;
        color: #FFFFFF !important;
        font-family: "SF Mono", "Roboto Mono", Menlo, Consolas, monospace !important; /* Force Mono */
    }
    
    /* Buttons */
    div.stButton > button {
        background-color: #3182F6;
        color: white !important;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.5rem 1.0rem;
        font-size: 14px;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background-color: #1B64DA;
        transform: translateY(-1px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #191F28;
        border-radius: 12px;
        padding: 6px 16px;
        font-weight: 600;
        color: #8B95A1 !important;
        border: 1px solid #333D4B;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3182F6 !important;
        color: #FFFFFF !important;
        border: 1px solid #3182F6;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #333D4B;
    }
    
    /* Status Badges */
    .badge-blue {
        background-color: rgba(49, 130, 246, 0.2);
        color: #3182F6;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)
