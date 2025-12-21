
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from pathlib import Path
import sys
import re

# Add parent dir to path
sys.path.append(str(Path(__file__).parent.parent))

from dashboard_shared import load_data, load_chart_data, load_css

st.set_page_config(
    page_title="Deep Analysis - Vibe Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

# Load Data
df_exp = load_data()

# --- Sidebar Search ---
st.sidebar.markdown("### 🔍 Strategy Search")
if not df_exp.empty:
    options = df_exp["search_label"].tolist()
    # If we have a selection from main page, find its index
    idx = 0
    if "selected_id" in st.session_state and st.session_state["selected_id"]:
        target_id = st.session_state["selected_id"]
        # Find matching string
        match = next((i for i, s in enumerate(options) if target_id in s), 0)
        idx = match
        
    sel_opt = st.sidebar.selectbox("Select Strategy", options, index=idx)
    
    # Update Session State from Sidebar
    if sel_opt:
        match = re.search(r'\((.{8})\)', sel_opt)
        if match:
            new_id = match.group(1)
            st.session_state["selected_id"] = new_id

# --- Main Content ---
st.markdown("# 🔍 Deep Analysis (심층 분석)")
if st.button("← Back to Dashboard"):
    st.switch_page("dashboard.py")

if df_exp.empty:
    st.info("No data available.")
    st.stop()

# Get Current Selection
selected_id = st.session_state.get("selected_id", None)
if not selected_id:
    # Default to best
    best_row = df_exp.sort_values("sharpe", ascending=False).iloc[0]
    selected_id = best_row["short_id"]

# Retrieve Row
row_slice = df_exp[df_exp["short_id"] == selected_id]
if row_slice.empty:
    st.error(f"Experiment ID {selected_id} not found.")
    st.stop()

row = row_slice.iloc[0]

# 1. Header Card
st.markdown(f"""
<div class="toss-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <span class="badge-blue">{row['origin_tag']} STRATEGY</span>
        <span style="color:#8B95A1;">ID: {row['id']}</span>
    </div>
    <h2 style="margin-top:10px; margin-bottom:5px;">{row['origin']} Strategy</h2>
    <p style="color:#B0B8C1;">{row['status']} | {row['fail_reason']}</p>
</div>
""", unsafe_allow_html=True)

# 2. Metrics & Config in 2 Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Performance Metrics (성과 지표)")
    st.markdown(f"""
    <div class="toss-card">
        <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:20px;">
            <div>
                <div style="font-size:12px; color:#8B95A1;">Sharpe Ratio</div>
                <div class="mono" style="font-size:24px; font-weight:700; color:#3182F6;">{row['sharpe']:.2f}</div>
            </div>
            <div>
                <div style="font-size:12px; color:#8B95A1;">Win Rate</div>
                <div class="mono" style="font-size:24px; font-weight:700; color:#FFFFFF;">{row['win_rate']*100:.1f}%</div>
            </div>
            <div>
                <div style="font-size:12px; color:#8B95A1;">Total Trades</div>
                <div class="mono" style="font-size:24px; font-weight:700; color:#FFFFFF;">{int(row['trades'])}</div>
            </div>
            <div>
                <div style="font-size:12px; color:#8B95A1;">Avg. Return</div>
                <div class="mono" style="font-size:24px; font-weight:700; color:#FFFFFF;">{row.get('return_mean', 0):.4f}</div>
            </div>
            <div>
                <div style="font-size:12px; color:#8B95A1;">Tot. Return</div>
                <div class="mono" style="font-size:24px; font-weight:700; color:#3182F6;">{row.get('total_return', 0):.2f}R</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🧬 Genome DNA (전략 유전자 구성)")
    
    # Initialize Registry for lookup
    from src.config import config
    from src.features.registry import FeatureRegistry
    
    @st.cache_resource
    def get_registry():
        r = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH))
        r.initialize()
        return r
    
    registry = get_registry()
    genome = row["genome_full"]
    
    if isinstance(genome, dict):
        for feature_id, params in genome.items():
            # Lookup metadata
            meta = registry.get(feature_id)
            meta_name = meta.name if meta else feature_id
            meta_desc = meta.description if meta else "Custom or Legacy Feature"
            category = meta.category.upper() if meta else "UNKNOWN"
            
            # Format Params
            param_html = ""
            for k, v in params.items():
                param_html += f'<span style="background:#333D4B; padding:2px 6px; border-radius:4px; margin-right:4px; font-size:11px; color:#B0B8C1;">{k}: <span style="color:#FFF;">{v}</span></span>'
            
            card_html = f"""
            <div style="background-color:#202632; padding:15px; border-radius:12px; border:1px solid #333D4B; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span style="color:#3182F6; font-weight:700; font-size:14px;">{meta_name}</span>
                    <span style="font-size:10px; color:#8B95A1; border:1px solid #4E5968; padding:2px 4px; border-radius:4px;">{category}</span>
                </div>
                <div style="font-size:12px; color:#B0B8C1; margin-bottom:10px;">{meta_desc}</div>
                <div style="font-family:monospace;">{param_html}</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.json(genome)

# 3. Charts & Analysis
st.divider()
st.markdown("### 📈 Deep Dive Analysis")

# A. Feature Importance
verdict = row.get("verdict")
if verdict and isinstance(verdict, dict) and "feature_importance" in verdict:
    st.markdown("#### 🧠 ML Guard: Feature Importance")
    imp_dict = verdict["feature_importance"]
    if imp_dict:
        df_imp = pd.DataFrame(list(imp_dict.items()), columns=["Feature", "Importance"])
        df_imp = df_imp.sort_values("Importance", ascending=True)
        
        fig_imp = px.bar(
            df_imp, x="Importance", y="Feature", orientation='h',
            title="What drove the ML decisions?",
            color="Importance", color_continuous_scale="Blues"
        )
        fig_imp.update_layout(
            plot_bgcolor="#191F28", paper_bgcolor="#191F28",
            xaxis=dict(showgrid=True, gridcolor="#333D4B", title="Gain Importance", tickfont=dict(color="#8B95A1")),
            yaxis=dict(title=None, tickfont=dict(color="#FFFFFF")),
            font=dict(color="#8B95A1"),
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("No feature importance data available.")

# B. Equity & Confidence
st.markdown("#### 💸 Equity Curve & ML Confidence")
df_chart = load_chart_data(row["id"])

if not df_chart.empty and "net_pnl" in df_chart.columns:
    df_chart["equity"] = df_chart["net_pnl"].cumsum()
    
    # Dual Subplot: Equity (Top) + ML Scale (Bottom)
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Strategy Equity", "ML Guard Confidence (Position Scale)")
    )
    
    # 1. Equity Line
    fig.add_trace(
        go.Scatter(x=df_chart.index, y=df_chart["equity"], name="Equity", line=dict(color="#3182F6", width=2)),
        row=1, col=1
    )
    
    # 2. ML Scale Bar/Area (if exists)
    if "scale" in df_chart.columns:
        # Filter for non-zero scales to avoid clutter
        df_scale = df_chart[df_chart["scale"] > 0]
        fig.add_trace(
            go.Bar(
                x=df_scale.index, y=df_scale["scale"], 
                name="Confidence", 
                marker_color="#00C4B4",
                opacity=0.6
            ),
            row=2, col=1
        )
    else:
        # Fallback if no scale (legacy)
        fig.add_annotation(text="No ML Confidence Data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, row=2, col=1)

    fig.update_layout(
        plot_bgcolor="#191F28", paper_bgcolor="#191F28",
        xaxis=dict(showgrid=False, tickfont=dict(color="#8B95A1", family="monospace")),
        yaxis=dict(showgrid=True, gridcolor="#333D4B", tickfont=dict(color="#8B95A1")),
        xaxis2=dict(showgrid=False, tickfont=dict(color="#8B95A1"), row=2, col=1),
        yaxis2=dict(showgrid=True, gridcolor="#333D4B", range=[0, 1.1], tickfont=dict(color="#8B95A1")),
        hovermode="x unified",
        margin=dict(l=0,r=0,t=20,b=20),
        height=500,
        legend=dict(orientation="h", y=1.05, x=0, font=dict(color="#8B95A1"))
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # C. Scatter: Returns vs Confidence (Validation of Stage 3)
    # Does higher confidence actually lead to higher returns?
    if "scale" in df_chart.columns:
        st.markdown("#### 🎯 ML Validation: Confidence vs. Trade PnL")
        
        # Filter for trade points (where pred != 0)
        # Note: df_chart is bar-by-bar PnL. We want per-trade PnL or at least bar PnL distribution.
        # Simple approach: Scatter of Bar PnL vs Scale (for active bars)
        df_active = df_chart[df_chart["pred"] != 0].copy()
        
        if not df_active.empty:
            corr = df_active["scale"].corr(df_active["net_pnl"])
            st.caption(f"Correlation (Confidence vs PnL): {corr:.4f}")
            
            fig_scatter = px.scatter(
                df_active, x="scale", y="net_pnl", 
                color="net_pnl", color_continuous_scale="RdBu",
                title="Are we sizing up on winners?",
                labels={"scale": "ML Confidence (Scale)", "net_pnl": "Bar PnL"}
            )
            fig_scatter.update_layout(
                plot_bgcolor="#191F28", paper_bgcolor="#191F28",
                xaxis=dict(showgrid=True, gridcolor="#333D4B"),
                yaxis=dict(showgrid=True, gridcolor="#333D4B"),
                font=dict(color="#8B95A1"),
                height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("No detailed chart data available for this experiment.")

# 4. JSON Dump
with st.expander("Show Raw Verdict Dump"):
    st.json(row["verdict"])
