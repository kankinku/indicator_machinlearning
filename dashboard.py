
import streamlit as st
import pandas as pd
import plotly.express as px
import time
from pathlib import Path
import sys

# Add current dir to path
sys.path.append(str(Path(__file__).parent))

from dashboard_shared import load_data, load_chart_data, load_css, render_sidebar

st.set_page_config(
    page_title="Vibe Trading Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()
render_sidebar()
df_exp = load_data()

# Header
c1, c2 = st.columns([0.8, 0.2])
with c1:
    st.markdown("# Vibe Intelligence Lab")
    st.markdown("<p class='small-text'>Autonomous Meta-RL Trading Agent Dashboard</p>", unsafe_allow_html=True)

with c2:
    if st.button("Refresh System (새로고침)"):
        st.rerun()

if df_exp.empty:
    st.info("System initializing... Waiting for first experiment results. (시스템 초기화 중...)")
    time.sleep(2)
    st.rerun()

# --- Top KPI Overlay ---
approved = df_exp[df_exp["status"] == "Approved"]
best_row = (
    approved.sort_values(
        ["holistic_score", "stability_pass", "total_return"],
        ascending=[False, False, False],
    ).iloc[0]
    if not approved.empty
    else None
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Total Experiments (총 실험)", len(df_exp))
with kpi2:
    app_count = len(df_exp[df_exp['status']=="Approved"])
    rate = (app_count/len(df_exp))*100 if len(df_exp)>0 else 0
    st.metric("Approval Rate (승인율)", f"{rate:.1f}%", f"{app_count} Approved (승인)")
with kpi3:
    if best_row is not None:
        st.metric("Best Model Sharpe (선정 샤프)", f"{best_row['sharpe']:.2f}")
    else:
        st.metric("Best Model Sharpe (선정 샤프)", "-")
with kpi4:
    if best_row is not None:
        st.metric("Best Model Win Rate (선정 승률)", f"{best_row['win_rate']*100:.1f}%")
    else:
        st.metric("Best Model Win Rate (선정 승률)", "-")
with kpi5:
    if best_row is not None:
        tot = best_row.get("total_return", 0)
        st.metric("Best Model Return (선정 수익)", f"{tot:.1f}%")
    else:
        st.metric("Best Model Return (선정 수익)", "-")

st.divider()

# 1. Best Model Highlight (Only if exists)
if best_row is not None:
    st.markdown("### 🏆 Hall of Fame")
    st.markdown(f"""
<div class="toss-card">
<div style="display:flex; justify-content:space-between; margin-bottom:10px;">
<span class="badge-blue">BEST PERFORMER (최고 성과)</span>
<span style="color:#8B95A1; font-size:12px;">{best_row['timestamp'].strftime('%Y-%m-%d %H:%M')}</span>
</div>
<h3 style="margin:0; color:#FFFFFF;">{best_row['origin']} Strategy (전략)</h3>
<p style="color:#B0B8C1; margin-top:4px; font-size:14px;">Composition (구성): {best_row['indicators']}</p>

<div style="display:grid; grid-template-columns: repeat(5, 1fr); gap:15px; margin-top:20px;">
<div>
<div style="font-size:12px; color:#8B95A1;">Sharpe (샤프)</div>
<div class="mono" style="font-size:24px; font-weight:700; color:#3182F6;">{best_row['sharpe']:.2f}</div>
</div>
<div>
<div style="font-size:12px; color:#8B95A1;">Win Rate (승률)</div>
<div class="mono" style="font-size:24px; font-weight:700; color:#FFFFFF;">{best_row['win_rate']*100:.1f}%</div>
</div>
<div>
<div style="font-size:12px; color:#8B95A1;">Trades (거래수)</div>
<div class="mono" style="font-size:24px; font-weight:700; color:#FFFFFF;">{int(best_row['trades'])}</div>
</div>
<div>
<div style="font-size:12px; color:#8B95A1;">Avg. Return (평균)</div>
<div class="mono" style="font-size:24px; font-weight:700; color:#FFFFFF;">{best_row.get('return_mean', 0):.2f}%</div>
</div>
<div>
<div style="font-size:12px; color:#8B95A1;">Tot. Return (누적)</div>
<div class="mono" style="font-size:24px; font-weight:700; color:#3182F6;">{best_row.get('total_return', 0):.2f}%</div>
</div>
</div>
<p style="color:#8B95A1; font-size:12px; margin-top:10px;">Risk: Target {best_row.get('target_return_pct', 0):.2f}% | Stop {best_row.get('stop_loss_pct', 0):.2f}% | Horizon {int(best_row.get('horizon', 0)) if pd.notna(best_row.get('horizon', 0)) else 0} | Profile {best_row.get('risk_profile', '-')} </p>
</div>
""", unsafe_allow_html=True)
    
    # Equity Curve for Best
    df_chart = load_chart_data(best_row["id"])
    if not df_chart.empty and "net_pnl" in df_chart.columns:
        stop_loss_pct = best_row.get('stop_loss_pct', 1.0)
        if pd.notna(stop_loss_pct):
            risk_unit = stop_loss_pct / 100.0
        else:
            risk_unit = 0.01
        returns = df_chart["net_pnl"].fillna(0.0) * risk_unit
        returns = returns.clip(lower=-0.95)
        df_chart["equity"] = (1.0 + returns).cumprod() - 1.0
        df_chart["equity"] = df_chart["equity"] * 100.0
        
        fig = px.area(df_chart, x=df_chart.index, y="equity", height=300)
        fig.update_layout(
            plot_bgcolor="#191F28", paper_bgcolor="#191F28",
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(showgrid=False, title=None, tickfont=dict(color="#8B95A1", family="monospace")),
            yaxis=dict(showgrid=True, gridcolor="#333D4B", title=None, tickfont=dict(color="#8B95A1", family="monospace")),
            hovermode="x unified"
        )
        fig.update_traces(line_color="#3182F6", fillcolor="rgba(49, 130, 246, 0.1)")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# 2. Experiment History (실험 기록)
st.subheader("Experiment History (실험 기록)")
st.info("💡 Click on any row to view Deep Analysis (행을 클릭하면 상세 분석 페이지로 이동합니다)")

# Filtering & Search
col_search, col_sort = st.columns([2, 1])
with col_search:
    search_q = st.text_input("🔍 Search Strategy (전략 검색)", placeholder="e.g. RSI, TREND")
with col_sort:
    sort_by = st.selectbox(
        "Sort By", 
        ["holistic_score", "timestamp", "sharpe", "win_rate", "total_return"],
        format_func=lambda x: x.replace("_", " ").title()
    )

# Filter Logic
df_display = df_exp.copy()
if search_q:
    df_display = df_display[df_display["indicators"].str.contains(search_q, case=False) | df_display["origin"].str.contains(search_q, case=False)]

df_display = df_display.sort_values(sort_by, ascending=False).reset_index(drop=True)

# Display standard dataframe with premium styling & SELECTION
selection = st.dataframe(
    df_display[[
        "timestamp", "generation", "origin", "sharpe", "win_rate", "trades", "total_return", "status", "short_id"
    ]],
    column_config={
        "timestamp": st.column_config.DatetimeColumn("Time (시간)", format="MM/DD HH:mm"),
        "generation": st.column_config.NumberColumn("Gen (세대)", format="#%d"),
        "origin": "Strategy (전략)",
        "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
        "win_rate": st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=1),
        "trades": "Trades",
        "total_return": st.column_config.NumberColumn("Return (%)", format="%.2f%%"),
        "status": "Status",
        "short_id": "ID"
    },
    use_container_width=True,
    hide_index=True,
    on_select="rerun",  # Triggers rerun on click
    selection_mode="single-row" 
)

# Handle Selection
if selection and selection["selection"]["rows"]:
    selected_idx = selection["selection"]["rows"][0]
    # Get ID from the DISPLAYED dataframe using the selected index
    selected_short_id = df_display.iloc[selected_idx]["short_id"]
    
    st.session_state["selected_id"] = selected_short_id
    st.switch_page("pages/01_Strategy_Analysis.py")

