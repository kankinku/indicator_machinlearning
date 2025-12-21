
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add parent dir to path
sys.path.append(str(Path(__file__).parent.parent))

from dashboard_shared import load_css
from src.config import config
from src.features.registry import FeatureRegistry

st.set_page_config(
    page_title="Feature Universe - Vibe Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

# --- Initialize Registry ---
@st.cache_resource
def get_registry():
    r = FeatureRegistry(str(config.FEATURE_REGISTRY_PATH))
    r.initialize()
    return r

registry = get_registry()

# --- Main Content ---
st.markdown("# 🧬 Feature Universe (지표 유니버스)")
st.markdown("""
<div class="toss-card">
    <h3 style="margin-top:0; color:white;">Evolving Feature Ecosystem</h3>
    <p style="color:#B0B8C1; font-size:14px;">
        Here lives the DNA of the trading strategies. The Evolutionary Agent selects and combines these genes (Indicators) 
        to adapt to the market regime.
        (여기는 전략의 DNA가 보관되는 곳입니다. 진화형 에이전트는 이 유전자들을 조합하여 시장 상황에 적응합니다.)
    </p>
</div>
""", unsafe_allow_html=True)

if st.button("← Back to Dashboard"):
    st.switch_page("dashboard.py")

st.divider()

# --- Stats ---
features = registry.list_all()
col1, col2, col3 = st.columns(3)
col1.metric("Total Genes (Indicators)", len(features))
col2.metric("Categories", len(set(f.category for f in features)))
col3.metric("Total Active Genomes", "0 (Initializing)") # Placeholder

st.divider()

# --- Filter ---
categories = sorted(list(set(f.category for f in features)))
selected_cat = st.radio("Filter by Category", ["ALL"] + categories, horizontal=True)

filtered_features = features
if selected_cat != "ALL":
    filtered_features = [f for f in features if f.category == selected_cat]

# --- Display Grid ---
# Use grid layout
n_cols = 3
rows = [filtered_features[i:i + n_cols] for i in range(0, len(filtered_features), n_cols)]

for row in rows:
    cols = st.columns(n_cols)
    for idx, item in enumerate(row):
        with cols[idx]:
            # Build Param String
            param_list = []
            for p in item.params:
                p_text = f"{p.name}"
                if p.default is not None:
                    p_text += f"={p.default}"
                param_list.append(p_text)
            
            param_str = ", ".join(param_list) if param_list else "No params"
            
            # Card HTML
            card_html = (
                f'<div class="toss-card" style="padding:16px; margin-bottom:12px; height:100%;">'
                f'    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">'
                f'        <span class="badge-blue" style="font-size:10px;">{item.feature_id}</span>'
                f'        <span style="font-size:10px; color:#6B7684;">{item.source}</span>'
                f'    </div>'
                f'    <div style="font-weight:700; color:white; font-size:16px; margin-bottom:4px;">{item.name}</div>'
                f'    <div style="font-size:13px; color:#8B95A1; margin-bottom:12px; line-height:1.4; min-height:40px;">{item.description}</div>'
                f'    <div style="background-color:#202632; padding:8px; border-radius:6px; border:1px solid #333D4B;">'
                f'        <div style="font-size:10px; color:#8B95A1; text-transform:uppercase; margin-bottom:2px;">Parameters</div>'
                f'        <div class="mono" style="color:#E5E8EB; font-size:11px; overflow-wrap:break-word;">{param_str}</div>'
                f'    </div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Code Viewer Expander
            with st.expander("View Code DNA"):
                st.code(item.code_snippet, language="python")
