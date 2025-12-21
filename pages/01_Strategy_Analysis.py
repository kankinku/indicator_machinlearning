
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path
import sys
import re

# Add parent dir to path
sys.path.append(str(Path(__file__).parent.parent))

from dashboard_shared import load_data, load_chart_data, load_css, render_sidebar

st.set_page_config(
    page_title="Deep Analysis - Vibe Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()
render_sidebar()

# Custom CSS for this page
st.markdown("""
<style>
.analysis-header {
    background: linear-gradient(135deg, #1A1F29 0%, #252B38 100%);
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 20px;
    border: 1px solid #333D4B;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 20px;
    margin-top: 25px;
}
.metric-item {
    background: rgba(255,255,255,0.03);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    min-height: 105px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border: 1px solid rgba(255,255,255,0.05);
}
.metric-label {
    font-size: 12px;
    color: #8B95A1;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    font-family: 'SF Mono', monospace;
}
.genome-card {
    background: #202632;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #333D4B;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}
.genome-card:hover {
    border-color: #3182F6;
    transform: translateY(-2px);
}
.section-title {
    font-size: 18px;
    font-weight: 600;
    color: #FFFFFF;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::before {
    content: '';
    width: 4px;
    height: 20px;
    background: #3182F6;
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)

# Load Data
df_exp = load_data()

if df_exp.empty:
    st.info("No data available.")
    st.stop()

# Get Current Selection
selected_id = st.session_state.get("selected_id", None)
if not selected_id:
    approved = df_exp[df_exp["status"] == "Approved"]
    base_df = approved if not approved.empty else df_exp
    best_row = base_df.sort_values(
        ["stability_pass", "total_return", "win_rate", "trades", "vol_ratio"],
        ascending=[False, False, False, False, True],
    ).iloc[0]
    selected_id = best_row["short_id"]

# Retrieve Row
row_slice = df_exp[df_exp["short_id"] == selected_id]
if row_slice.empty:
    st.error(f"Experiment ID {selected_id} not found.")
    st.stop()

row = row_slice.iloc[0]

# --- Navigation ---
col_back, col_title, col_nav = st.columns([1, 6, 3])
with col_back:
    if st.button("← Dashboard"):
        st.switch_page("dashboard.py")

with col_title:
    st.markdown("# 🔍 Deep Analysis")

with col_nav:
    # Quick Navigation
    options = df_exp["short_id"].tolist()
    labels = [f"{r['origin']} ({r['short_id']})" for _, r in df_exp.iterrows()]
    current_idx = options.index(selected_id) if selected_id in options else 0
    
    new_sel = st.selectbox("Quick Jump (빠른 이동)", options, index=current_idx, format_func=lambda x: labels[options.index(x)])
    if new_sel != selected_id:
        st.session_state["selected_id"] = new_sel
        st.rerun()

# --- 1. Header Card ---
status_color = "#3182F6" if row['status'] == "Approved" else "#F04452"
sharpe_color = "#3182F6" if row['sharpe'] > 1.0 else ("#00C4B4" if row['sharpe'] > 0 else "#F04452")

st.markdown(f"""
<div class="analysis-header">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
<div>
<span style="background:{status_color}22; color:{status_color}; padding:6px 16px; border-radius:20px; font-size:12px; font-weight:600;">
{row['status'].upper()}
</span>
<span style="color:#8B95A1; margin-left:15px; font-size:12px;">Gen #{row['generation']} | ID: {row['id'][:8]}...</span>
</div>
<span style="color:#8B95A1; font-size:13px;">{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</span>
</div>

<h1 style="margin:0; font-size:32px; color:#FFFFFF;">{row['origin']} Strategy</h1>
<p style="color:#B0B8C1; margin-top:8px; font-size:14px;">
{row['indicators']}
</p>

<div class="metric-grid">
<div class="metric-item">
<div class="metric-label">Sharpe Ratio <span style="font-size:10px; opacity:0.7;">(샤프 지수)</span></div>
<div class="metric-value" style="color:{sharpe_color};">{row['sharpe']:.2f}</div>
</div>
<div class="metric-item">
<div class="metric-label">Win Rate <span style="font-size:10px; opacity:0.7;">(승률)</span></div>
<div class="metric-value" style="color:#FFFFFF;">{row['win_rate']*100:.1f}%</div>
</div>
<div class="metric-item">
<div class="metric-label">Total Trades <span style="font-size:10px; opacity:0.7;">(총 거래수)</span></div>
<div class="metric-value" style="color:#FFFFFF;">{int(row['trades'])}</div>
</div>
<div class="metric-item">
<div class="metric-label">Avg Return <span style="font-size:10px; opacity:0.7;">(평균 수익)</span></div>
<div class="metric-value" style="color:#FFFFFF;">{row.get('return_mean', 0):.4f}</div>
</div>
<div class="metric-item">
<div class="metric-label">Cumulative <span style="font-size:10px; opacity:0.7;">(누적 수익)</span></div>
<div class="metric-value" style="color:#3182F6;">{row.get('total_return', 0):.2f}R</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# --- 2. Two Column Layout: Genome + Charts ---
col_genome, col_charts = st.columns([1, 2])

with col_genome:
    st.markdown("<div class='section-title'>Strategy DNA</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 11px; color: #8B95A1; margin-top: -15px; margin-bottom: 20px;'>에이전트가 조합한 전략의 논리적 구성 요소와 파라미터입니다.</div>", unsafe_allow_html=True)
    
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
    
    if isinstance(genome, dict) and genome:
        # Build one big HTML string for the scroll container
        genome_html = '<div class="genome-scroll-container">'
        
        for feature_id, params in genome.items():
            meta = registry.get(feature_id)
            meta_name = meta.name if meta else feature_id
            meta_desc = meta.description if meta else "Custom Feature"
            category = meta.category.upper() if meta else "CUSTOM"
            
            # Category Color
            cat_colors = {
                "MOMENTUM": "#F04452",
                "TREND": "#3182F6",
                "VOLATILITY": "#FFA500",
                "VOLUME": "#00C4B4",
                "CONTEXT": "#9B59B6",
                "CUSTOM": "#8B95A1"
            }
            cat_color = cat_colors.get(category, "#8B95A1")
            
            # Params
            param_chips = ""
            for k, v in params.items():
                param_chips += f'<span style="background:#282F3A; padding:4px 10px; border-radius:6px; margin-right:6px; font-size:11px; color:#E5E8EB; font-family:monospace; border:1px solid #333D4B;">{k}={v}</span>'
            
            card_html = f"""
<div class="genome-card">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
<span style="color:#FFFFFF; font-weight:600; font-size:15px;">{meta_name}</span>
<span style="background:{cat_color}22; color:{cat_color}; padding:3px 10px; border-radius:10px; font-size:10px; font-weight:600;">{category}</span>
</div>
<div style="font-size:12px; color:#8B95A1; margin-bottom:12px;">{meta_desc}</div>
<div style="display:flex; flex-wrap:wrap; gap:4px;">{param_chips if param_chips else '<span style="color:#4E5968; font-size:11px;">No parameters</span>'}</div>
</div>
"""
            genome_html += card_html
            
        genome_html += '</div>'
        st.markdown(genome_html, unsafe_allow_html=True)
    else:
        st.info("No genome data available")



with col_charts:
    st.markdown("<div class='section-title'>Performance Charts</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 11px; color: #8B95A1; margin-top: -15px; margin-bottom: 20px;'>누적 수익률 곡선과 모델의 거래 시점별 확신도(Confidence) 추이입니다.</div>", unsafe_allow_html=True)
    
    df_chart = load_chart_data(row["id"])
    
    if not df_chart.empty and "net_pnl" in df_chart.columns:
        df_chart["equity"] = df_chart["net_pnl"].cumsum()
        
        # Dual Chart: Equity + ML Confidence
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08,
            row_heights=[0.65, 0.35],
            subplot_titles=("📈 Equity Curve", "🎯 ML Confidence Score")
        )
        
        # Equity Line
        fig.add_trace(
            go.Scatter(
                x=df_chart.index, y=df_chart["equity"], 
                name="Equity",
                line=dict(color="#3182F6", width=2),
                fill="tozeroy",
                fillcolor="rgba(49, 130, 246, 0.1)"
            ),
            row=1, col=1
        )
        
        # ML Scale
        if "scale" in df_chart.columns:
            df_scale = df_chart[df_chart["scale"] > 0]
            fig.add_trace(
                go.Bar(
                    x=df_scale.index, y=df_scale["scale"], 
                    name="Confidence",
                    marker_color="#00C4B4",
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            plot_bgcolor="#191F28", paper_bgcolor="#191F28",
            xaxis=dict(showgrid=False, tickfont=dict(color="#8B95A1")),
            yaxis=dict(showgrid=True, gridcolor="#333D4B", tickfont=dict(color="#8B95A1"), title="PnL (R)"),
            xaxis2=dict(showgrid=False, tickfont=dict(color="#8B95A1")),
            yaxis2=dict(showgrid=True, gridcolor="#333D4B", range=[0, 1.1], tickfont=dict(color="#8B95A1"), title="Scale"),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            height=450,
            legend=dict(orientation="h", y=1.12, x=0, font=dict(color="#8B95A1")),
            font=dict(color="#8B95A1")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No chart data available for this experiment.")

# --- 3. Feature Importance ---
st.divider()
st.markdown("<div class='section-title'>ML Feature Importance</div>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 11px; color: #8B95A1; margin-top: -15px; margin-bottom: 20px;'>AI 모델이 진입/청산 결정을 내릴 때 가장 중요하게 참고한 지표들입니다.</div>", unsafe_allow_html=True)

verdict = row.get("verdict")
if verdict and isinstance(verdict, dict) and "feature_importance" in verdict:
    imp_dict = verdict["feature_importance"]
    if imp_dict:
        df_imp = pd.DataFrame(list(imp_dict.items()), columns=["Feature", "Importance"])
        df_imp = df_imp.sort_values("Importance", ascending=True).tail(15)  # Top 15
        
        fig_imp = px.bar(
            df_imp, x="Importance", y="Feature", orientation='h',
            color="Importance", color_continuous_scale=["#333D4B", "#3182F6"]
        )
        fig_imp.update_layout(
            plot_bgcolor="#191F28", paper_bgcolor="#191F28",
            xaxis=dict(showgrid=True, gridcolor="#333D4B", title="Gain Importance", tickfont=dict(color="#8B95A1")),
            yaxis=dict(title=None, tickfont=dict(color="#FFFFFF", size=11)),
            font=dict(color="#8B95A1"),
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("Feature importance data not available for this experiment.")

# --- 4. Validation Scatter ---
if not df_chart.empty and "scale" in df_chart.columns:
    st.divider()
    st.markdown("<div class='section-title'>ML Validation: Confidence vs Return</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 11px; color: #8B95A1; margin-top: -15px; margin-bottom: 20px;'>AI의 확신도(Confidence)와 실제 수익 사이의 상관관계입니다. 양의 상관관계(Positive)가 높을수록 믿을 수 있는 모델입니다.</div>", unsafe_allow_html=True)
    
    df_active = df_chart[df_chart["pred"] != 0].copy()
    
    if not df_active.empty and len(df_active) > 5:
        corr = df_active["scale"].corr(df_active["net_pnl"])
        corr_color = "#00C4B4" if corr > 0.1 else ("#F04452" if corr < -0.1 else "#8B95A1")
        
        col_corr, col_chart = st.columns([1, 3])
        with col_corr:
            st.markdown(f"""
            <div style="background:#202632; padding:25px; border-radius:16px; text-align:center;">
                <div style="font-size:12px; color:#8B95A1; margin-bottom:10px;">CORRELATION</div>
                <div style="font-size:36px; font-weight:700; color:{corr_color}; font-family:monospace;">{corr:.3f}</div>
                <div style="font-size:11px; color:#8B95A1; margin-top:10px;">
                    {'✅ Positive: Higher confidence = Better trades' if corr > 0.1 else '⚠️ Weak or Negative correlation'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_chart:
            fig_scatter = px.scatter(
                df_active, x="scale", y="net_pnl", 
                color="net_pnl", color_continuous_scale="RdBu",
                labels={"scale": "ML Confidence", "net_pnl": "Trade PnL"}
            )
            fig_scatter.update_layout(
                plot_bgcolor="#191F28", paper_bgcolor="#191F28",
                xaxis=dict(showgrid=True, gridcolor="#333D4B", tickfont=dict(color="#8B95A1")),
                yaxis=dict(showgrid=True, gridcolor="#333D4B", tickfont=dict(color="#8B95A1")),
                font=dict(color="#8B95A1"),
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

# --- 5. Raw Data ---
with st.expander("📋 Raw Verdict Data (원본 데이터)"):
    st.json(row["verdict"])
