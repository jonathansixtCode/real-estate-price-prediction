"""
Streamlit Property Valuation Tool for Real Estate Agents & Buyers
Run from project root: streamlit run app/streamlit_app.py
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.deployment import predict_with_confidence
except ImportError:
    from deployment import predict_with_confidence

st.set_page_config(
    page_title="Ames Property Valuation Studio",
    page_icon="🏠",
    layout="wide",
)


def resolve_bundle_path() -> str:
    env_path = os.getenv("MODEL_BUNDLE_PATH")
    candidates = [p for p in [env_path, "models/ames_housing_bundle.joblib", os.path.join(PROJECT_ROOT, "models/ames_housing_bundle.joblib"), os.path.join(HERE, "..", "models/ames_housing_bundle.joblib")] if p]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return candidates[0] if candidates else "{bundle_rel_path}"


BUNDLE_PATH = resolve_bundle_path()


@st.cache_resource
def load_bundle(path: str):
    if not path or not os.path.exists(path):
        return None
    return joblib.load(path)


bundle = load_bundle(BUNDLE_PATH)

st.title("🏠 Ames Property Valuation Studio")
st.markdown(
    "Interactive valuation tool for real estate agents and home buyers. "
    "Adjust key property drivers below to obtain an instant price estimate and calibrated confidence range."
)

if bundle is None:
    st.error(
        f"Model bundle not found at `{BUNDLE_PATH}`. "
        "Please run Module 6 (`Module_6_Deployment.ipynb` or `python src/deployment.py`) first."
    )
    st.stop()

feature_names = bundle["feature_names"]
medians = bundle["feature_medians"]
mins = bundle["feature_mins"]
maxs = bundle["feature_maxs"]

priority_candidates = [
    "OverallQual",
    "Qual_x_TotalSF",
    "TotalSF",
    "GrLivArea",
    "Neighborhood_Median_PricePerSqM",
    "TotalBathrooms",
    "GarageCars",
    "PropertyAge",
    "YearsSinceRemodel",
    "TotalBsmtSF",
    "Qual_x_GrLivArea",
    "AmenityProximityScore",
]
active_sliders = [f for f in priority_candidates if f in feature_names]
if len(active_sliders) < 8:
    for f in feature_names:
        if f not in active_sliders:
            active_sliders.append(f)
        if len(active_sliders) >= 10:
            break

st.sidebar.header("Valuation Settings")
conf_level = st.sidebar.select_slider(
    "Confidence Interval Level",
    options=[0.80, 0.90, 0.95, 0.99],
    value=0.95,
    format_func=lambda x: f"{int(x * 100)}%",
)
m_type = bundle["model_type"]
cv_rmse = bundle["cv_rmse_log"]
n_feat = len(feature_names)
st.sidebar.info(f"**Model:** {m_type}\n\n**5-Fold CV RMSE (log):** {cv_rmse:.4f}\n\n**Features Used:** {n_feat}")

st.subheader("1. Configure Key Property Attributes (Scaled Feature Space)")
st.caption("Sliders are initialized at the Ames market median (0.0 in RobustScaler units).")

user_inputs = dict(medians)
cols = st.columns(3)

for idx, feat in enumerate(active_sliders):
    col = cols[idx % 3]
    low = float(min(mins.get(feat, -3.0), -2.0))
    high = float(max(maxs.get(feat, 3.0), 2.0))
    default_val = float(medians.get(feat, 0.0))
    step = max((high - low) / 100.0, 0.01)
    user_inputs[feat] = col.slider(
        label=feat,
        min_value=round(low, 2),
        max_value=round(high, 2),
        value=round(default_val, 2),
        step=round(step, 2),
    )

result = predict_with_confidence(
    bundle=bundle,
    input_features=user_inputs,
    confidence_level=conf_level,
)

est = result["estimated_price_usd"]
low_b = result["confidence_range_usd"]["lower_bound"]
high_b = result["confidence_range_usd"]["upper_bound"]

st.divider()
st.subheader("2. Estimated Market Valuation")

m1, m2, m3 = st.columns(3)
m1.metric("Estimated Sale Price", f"${est:,.0f}")
m2.metric(f"Lower Bound ({int(conf_level*100)}% CI)", f"${low_b:,.0f}", delta=f"-${est - low_b:,.0f}", delta_color="off")
m3.metric(f"Upper Bound ({int(conf_level*100)}% CI)", f"${high_b:,.0f}", delta=f"+${high_b - est:,.0f}", delta_color="off")

range_df = pd.DataFrame({
    "Tier": ["Conservative (Lower CI)", "Point Estimate", "Optimistic (Upper CI)"],
    "Valuation ($ USD)": [low_b, est, high_b],
})
st.bar_chart(range_df.set_index("Tier"))
