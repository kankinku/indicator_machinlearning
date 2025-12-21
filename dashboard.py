
import streamlit as st
import pandas as pd
import plotly.express as px
import time
from pathlib import Path
import sys

# Add current dir to path
sys.path.append(str(Path(__file__).parent))

from dashboard_shared import load_data, load_chart_data, load_css

st.set_page_config(
    page_title="Vibe Trading Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()
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
best_row = df_exp[df_exp["status"]=="Approved"].sort_values("sharpe", ascending=False).iloc[0] if not df_exp[df_exp["status"]=="Approved"].empty else None

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Total Experiments (총 실험)", len(df_exp))
with kpi2:
    app_count = len(df_exp[df_exp['status']=="Approved"])
    rate = (app_count/len(df_exp))*100 if len(df_exp)>0 else 0
    st.metric("Approval Rate (승인율)", f"{rate:.1f}%", f"{app_count} Approved (승인)")
with kpi3:
    if best_row is not None:
        st.metric("Top Sharpe (최고 샤프)", f"{best_row['sharpe']:.2f}")
    else:
        st.metric("Top Sharpe (최고 샤프)", "-")
with kpi4:
    if best_row is not None:
        st.metric("Top Win Rate (최고 승률)", f"{best_row['win_rate']*100:.1f}%")
    else:
        st.metric("Top Win Rate (최고 승률)", "-")
with kpi5:
    if best_row is not None:
        # Check if total_return column exists (backward compatibility)
        if "total_return" in best_row:
             tot = best_row["total_return"]
        else:
             tot = best_row.get("return_mean", 0) * best_row.get("trades", 0)
             
        # Estimate % based on 5% Risk per Trade (Aggressive)
        est_pct = tot * 5.0 
        st.metric("Top Return (최고 수익)", f"{tot:.1f}R", f"Est. +{est_pct:.0f}%")
    else:
         st.metric("Top Return (최고 수익)", "-")

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
        <div class="mono" style="font-size:24px; font-weight:700; color:#FFFFFF;">{best_row.get('return_mean', best_row.get('return', 0)):.2f}</div>
    </div>
    <div>
        <div style="font-size:12px; color:#8B95A1;">Tot. Return (누적)</div>
        <div class="mono" style="font-size:24px; font-weight:700; color:#3182F6;">{best_row.get('total_return', 0):.2f}R</div>
    </div>
</div>
</div>
""", unsafe_allow_html=True)
    
    # Equity Curve for Best
    df_chart = load_chart_data(best_row["id"])
    if not df_chart.empty and "net_pnl" in df_chart.columns:
        df_chart["equity"] = df_chart["net_pnl"].cumsum()
        
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

# 2. Leaderboard Table with Selection
st.subheader("Experiment History (실험 기록)")
st.info("💡 Click on a row to view details (행을 클릭하면 상세 페이지로 이동합니다)")

# Simple Table structure
disp_df = df_exp[["short_id", "timestamp", "origin", "sharpe", "win_rate", "trades", "total_return", "status", "fail_reason"]].copy()
disp_df["win_rate_fmt"] = disp_df["win_rate"].apply(lambda x: f"{x*100:.1f}%")
disp_df["sharpe_fmt"] = disp_df["sharpe"].apply(lambda x: f"{x:.2f}")

event = st.dataframe(
    disp_df,
    column_config={
        "short_id": None, # Hide ID
        "timestamp": "Time (시간)",
        "origin": "Strategy (전략)",
        "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
        "sharpe_fmt": None, # Hide formatted helper
        "win_rate": None, # Hide raw
        "win_rate_fmt": "Win Rate (승률)",
        "trades": "Trades (거래)",
        "total_return": st.column_config.NumberColumn(
            "Cum. Return (누적수익)",
            format="%.2f R"
        ),
        "status": "Verdict (결과)",
        "fail_reason": "Note (비고)"
    },
    column_order=["timestamp", "origin", "sharpe", "win_rate_fmt", "trades", "total_return", "status", "fail_reason"],
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row"
)

# Handle Selection
if event.selection.rows:
    selected_idx = event.selection.rows[0]
    # Use the value from the displayed dataframe's specific row
    selected_id = disp_df.iloc[selected_idx]["short_id"]
    st.session_state["selected_id"] = selected_id
    st.switch_page("pages/01_Strategy_Analysis.py")

