"""
Module 6: Model Packaging, REST API & Interactive Demo Deployment
Ames Housing Dataset — Production Inference & Deployment Utilities

Provides modular functions for:
- Training and serializing a complete inference bundle (model + feature schema + CV residual std)
- Predicting house prices with calibrated 95% prediction/confidence intervals
- Generating production-ready FastAPI (`api.py`) and Streamlit (`streamlit_app.py`) scripts
- Generating cloud deployment artifacts (`Dockerfile`, `requirements_deploy.txt`, `render.yaml`)
"""

from typing import Dict, List, Any
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_best_estimator(random_state: int = 42) -> Any:
    """
    Return the top-performing regressor architecture from Module 5 (LightGBM primary,
    with XGBoost and RandomForest fallbacks).
    """
    if HAS_LGBM:
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample_for_bin=50000,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    if HAS_XGB:
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )


def build_and_save_model_bundle(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    artifact_dir: str = os.path.join("models"),
    bundle_filename: str = "ames_housing_bundle.joblib",
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Fit the production regressor, compute out-of-fold log-residual standard deviation
    and empirical quantiles for calibrated confidence intervals, and serialize the
    complete artifact bundle to disk.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    estimator = get_best_estimator(random_state=random_state)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X_train), dtype=float)

    for tr_idx, val_idx in kf.split(X_train):
        m = clone(estimator)
        m.fit(X_train.iloc[tr_idx], y_train_log.iloc[tr_idx])
        oof_preds[val_idx] = m.predict(X_train.iloc[val_idx])

    residuals_log = np.asarray(y_train_log) - oof_preds
    residual_std_log = float(np.std(residuals_log, ddof=1))
    q_low_log = float(np.quantile(residuals_log, 0.025))
    q_high_log = float(np.quantile(residuals_log, 0.975))
    cv_rmse_log = float(np.sqrt(np.mean(residuals_log ** 2)))

    final_model = clone(estimator)
    final_model.fit(X_train, y_train_log)

    feature_names = list(X_train.columns)
    feature_medians = X_train.median().to_dict()
    feature_mins = X_train.min().to_dict()
    feature_maxs = X_train.max().to_dict()

    bundle = {
        "model": final_model,
        "model_type": type(final_model).__name__,
        "feature_names": feature_names,
        "feature_medians": feature_medians,
        "feature_mins": feature_mins,
        "feature_maxs": feature_maxs,
        "residual_std_log": residual_std_log,
        "empirical_q025_log": q_low_log,
        "empirical_q975_log": q_high_log,
        "cv_rmse_log": cv_rmse_log,
    }

    bundle_path = os.path.join(artifact_dir, bundle_filename)
    joblib.dump(bundle, bundle_path)

    metadata = {
        "bundle_path": bundle_path,
        "model_type": bundle["model_type"],
        "n_features": len(feature_names),
        "cv_rmse_log": round(cv_rmse_log, 5),
        "residual_std_log": round(residual_std_log, 5),
        "empirical_95_interval_log": [round(q_low_log, 5), round(q_high_log, 5)],
    }

    meta_path = os.path.join(artifact_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def predict_with_confidence(
    bundle: Dict[str, Any],
    input_features: Dict[str, float],
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Generate a point price estimate (USD) and calibrated confidence/prediction range
    from a partial or complete dictionary of scaled features.
    """
    feature_names: List[str] = bundle["feature_names"]
    medians: Dict[str, float] = bundle["feature_medians"]
    model = bundle["model"]
    std_log: float = bundle["residual_std_log"]

    row = {col: float(input_features.get(col, medians.get(col, 0.0))) for col in feature_names}
    X_row = pd.DataFrame([row], columns=feature_names)

    pred_log = float(model.predict(X_row)[0])

    z_map = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    z = z_map.get(round(confidence_level, 2), 1.9600)

    lower_log = pred_log - z * std_log
    upper_log = pred_log + z * std_log

    estimate_usd = float(np.expm1(pred_log))
    lower_usd = float(np.expm1(lower_log))
    upper_usd = float(np.expm1(upper_log))

    return {
        "estimated_price_usd": round(estimate_usd, 2),
        "confidence_level": confidence_level,
        "confidence_range_usd": {
            "lower_bound": round(lower_usd, 2),
            "upper_bound": round(upper_usd, 2),
        },
        "log_prediction": round(pred_log, 5),
        "model_type": bundle.get("model_type", "Regressor"),
    }


def write_deployment_files(
    output_dir: str = "app",
    bundle_rel_path: str = "models/ames_housing_bundle.joblib",
) -> Dict[str, str]:
    """
    Generate standalone FastAPI service (`api.py`), interactive Streamlit app (`streamlit_app.py`),
    cloud dependencies (`requirements_deploy.txt`), `Dockerfile`, and `render.yaml`
    inside `output_dir` (default: `app`).
    """
    os.makedirs(output_dir, exist_ok=True)
    created: Dict[str, str] = {}

    api_lines = [
        '"""',
        'FastAPI Property Valuation Service — Ames Housing Price Estimator',
        'Run from project root: uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload',
        '"""',
        '',
        'from typing import Dict',
        'import os',
        'import sys',
        'import joblib',
        'from fastapi import FastAPI, HTTPException',
        'from fastapi.responses import RedirectResponse',
        'from pydantic import BaseModel, Field',
        '',
        'HERE = os.path.dirname(os.path.abspath(__file__))',
        'PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))',
        'for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):',
        '    if os.path.exists(p) and p not in sys.path:',
        '        sys.path.insert(0, p)',
        '',
        'try:',
        '    from src.deployment import predict_with_confidence',
        'except ImportError:',
        '    from deployment import predict_with_confidence',
        '',
        '',
        'def resolve_bundle_path() -> str:',
        '    env_path = os.getenv("MODEL_BUNDLE_PATH")',
        f'    candidates = [p for p in [env_path, "{bundle_rel_path}", os.path.join(PROJECT_ROOT, "{bundle_rel_path}"), os.path.join(HERE, "..", "{bundle_rel_path}")] if p]',
        '    for cand in candidates:',
        '        if os.path.exists(cand):',
        '            return cand',
        '    return candidates[0] if candidates else "{bundle_rel_path}"',
        '',
        '',
        'BUNDLE_PATH = resolve_bundle_path()',
        '',
        'app = FastAPI(',
        '    title="Ames Real Estate Valuation API",',
        '    description="Predicts residential property sale prices with calibrated confidence intervals.",',
        '    version="1.0.0",',
        ')',
        '',
        'model_bundle = None',
        '',
        '',
        '@app.on_event("startup")',
        'def load_model() -> None:',
        '    global model_bundle',
        '    path = resolve_bundle_path()',
        '    if not os.path.exists(path):',
        '        raise RuntimeError(f"Model bundle not found at {path}. Run Module 6 packaging first.")',
        '    model_bundle = joblib.load(path)',
        '',
        '',
        'class ValuationRequest(BaseModel):',
        '    features: Dict[str, float] = Field(',
        '        default_factory=dict,',
        '        description="Dictionary of scaled feature values (unspecified features default to training medians).",',
        '    )',
        '    confidence_level: float = Field(',
        '        default=0.95,',
        '        ge=0.50,',
        '        le=0.99,',
        '        description="Confidence level for prediction interval (e.g., 0.90 or 0.95).",',
        '    )',
        '',
        '',
        'class ConfidenceRange(BaseModel):',
        '    lower_bound: float',
        '    upper_bound: float',
        '',
        '',
        'class ValuationResponse(BaseModel):',
        '    estimated_price_usd: float',
        '    confidence_level: float',
        '    confidence_range_usd: ConfidenceRange',
        '    log_prediction: float',
        '    model_type: str',
        '',
        '',
        '@app.get("/", include_in_schema=False)',
        'def root_redirect() -> RedirectResponse:',
        '    return RedirectResponse(url="/docs")',
        '',
        '',
        '@app.get("/health")',
        'def health_check() -> Dict[str, str]:',
        '    return {"status": "healthy", "model_loaded": str(model_bundle is not None)}',
        '',
        '',
        '@app.get("/features")',
        'def list_features() -> Dict[str, object]:',
        '    if model_bundle is None:',
        '        raise HTTPException(status_code=503, detail="Model bundle not loaded.")',
        '    return {',
        '        "n_features": len(model_bundle["feature_names"]),',
        '        "feature_names": model_bundle["feature_names"],',
        '        "feature_medians": model_bundle["feature_medians"],',
        '    }',
        '',
        '',
        '@app.post("/predict", response_model=ValuationResponse)',
        'def predict_price(payload: ValuationRequest) -> ValuationResponse:',
        '    if model_bundle is None:',
        '        raise HTTPException(status_code=503, detail="Model bundle not loaded.")',
        '    result = predict_with_confidence(',
        '        bundle=model_bundle,',
        '        input_features=payload.features,',
        '        confidence_level=payload.confidence_level,',
        '    )',
        '    return ValuationResponse(**result)',
    ]
    api_path = os.path.join(output_dir, "api.py")
    with open(api_path, "w", encoding="utf-8") as f:
        f.write("\n".join(api_lines) + "\n")
    created["api.py"] = api_path

    st_lines = [
        '"""',
        'Streamlit Property Valuation Tool for Real Estate Agents & Buyers',
        'Run from project root: streamlit run app/streamlit_app.py',
        '"""',
        '',
        'import os',
        'import sys',
        'import joblib',
        'import pandas as pd',
        'import streamlit as st',
        '',
        'HERE = os.path.dirname(os.path.abspath(__file__))',
        'PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))',
        'for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):',
        '    if os.path.exists(p) and p not in sys.path:',
        '        sys.path.insert(0, p)',
        '',
        'try:',
        '    from src.deployment import predict_with_confidence',
        'except ImportError:',
        '    from deployment import predict_with_confidence',
        '',
        'st.set_page_config(',
        '    page_title="Ames Property Valuation Studio",',
        '    page_icon="🏠",',
        '    layout="wide",',
        ')',
        '',
        '',
        'def resolve_bundle_path() -> str:',
        '    env_path = os.getenv("MODEL_BUNDLE_PATH")',
        f'    candidates = [p for p in [env_path, "{bundle_rel_path}", os.path.join(PROJECT_ROOT, "{bundle_rel_path}"), os.path.join(HERE, "..", "{bundle_rel_path}")] if p]',
        '    for cand in candidates:',
        '        if os.path.exists(cand):',
        '            return cand',
        '    return candidates[0] if candidates else "{bundle_rel_path}"',
        '',
        '',
        'BUNDLE_PATH = resolve_bundle_path()',
        '',
        '',
        '@st.cache_resource',
        'def load_bundle(path: str):',
        '    if not path or not os.path.exists(path):',
        '        return None',
        '    return joblib.load(path)',
        '',
        '',
        'bundle = load_bundle(BUNDLE_PATH)',
        '',
        'st.title("🏠 Ames Property Valuation Studio")',
        'st.markdown(',
        '    "Interactive valuation tool for real estate agents and home buyers. "',
        '    "Adjust key property drivers below to obtain an instant price estimate and calibrated confidence range."',
        ')',
        '',
        'if bundle is None:',
        '    st.error(',
        '        f"Model bundle not found at `{BUNDLE_PATH}`. "',
        '        "Please run Module 6 (`Module_6_Deployment.ipynb` or `python src/deployment.py`) first."',
        '    )',
        '    st.stop()',
        '',
        'feature_names = bundle["feature_names"]',
        'medians = bundle["feature_medians"]',
        'mins = bundle["feature_mins"]',
        'maxs = bundle["feature_maxs"]',
        '',
        'priority_candidates = [',
        '    "OverallQual",',
        '    "Qual_x_TotalSF",',
        '    "TotalSF",',
        '    "GrLivArea",',
        '    "Neighborhood_Median_PricePerSqM",',
        '    "TotalBathrooms",',
        '    "GarageCars",',
        '    "PropertyAge",',
        '    "YearsSinceRemodel",',
        '    "TotalBsmtSF",',
        '    "Qual_x_GrLivArea",',
        '    "AmenityProximityScore",',
        ']',
        'active_sliders = [f for f in priority_candidates if f in feature_names]',
        'if len(active_sliders) < 8:',
        '    for f in feature_names:',
        '        if f not in active_sliders:',
        '            active_sliders.append(f)',
        '        if len(active_sliders) >= 10:',
        '            break',
        '',
        'st.sidebar.header("Valuation Settings")',
        'conf_level = st.sidebar.select_slider(',
        '    "Confidence Interval Level",',
        '    options=[0.80, 0.90, 0.95, 0.99],',
        '    value=0.95,',
        '    format_func=lambda x: f"{int(x * 100)}%",',
        ')',
        'm_type = bundle["model_type"]',
        'cv_rmse = bundle["cv_rmse_log"]',
        'n_feat = len(feature_names)',
        'st.sidebar.info(f"**Model:** {m_type}\\n\\n**5-Fold CV RMSE (log):** {cv_rmse:.4f}\\n\\n**Features Used:** {n_feat}")',
        '',
        'st.subheader("1. Configure Key Property Attributes (Scaled Feature Space)")',
        'st.caption("Sliders are initialized at the Ames market median (0.0 in RobustScaler units).")',
        '',
        'user_inputs = dict(medians)',
        'cols = st.columns(3)',
        '',
        'for idx, feat in enumerate(active_sliders):',
        '    col = cols[idx % 3]',
        '    low = float(min(mins.get(feat, -3.0), -2.0))',
        '    high = float(max(maxs.get(feat, 3.0), 2.0))',
        '    default_val = float(medians.get(feat, 0.0))',
        '    step = max((high - low) / 100.0, 0.01)',
        '    user_inputs[feat] = col.slider(',
        '        label=feat,',
        '        min_value=round(low, 2),',
        '        max_value=round(high, 2),',
        '        value=round(default_val, 2),',
        '        step=round(step, 2),',
        '    )',
        '',
        'result = predict_with_confidence(',
        '    bundle=bundle,',
        '    input_features=user_inputs,',
        '    confidence_level=conf_level,',
        ')',
        '',
        'est = result["estimated_price_usd"]',
        'low_b = result["confidence_range_usd"]["lower_bound"]',
        'high_b = result["confidence_range_usd"]["upper_bound"]',
        '',
        'st.divider()',
        'st.subheader("2. Estimated Market Valuation")',
        '',
        'm1, m2, m3 = st.columns(3)',
        'm1.metric("Estimated Sale Price", f"${est:,.0f}")',
        'm2.metric(f"Lower Bound ({int(conf_level*100)}% CI)", f"${low_b:,.0f}", delta=f"-${est - low_b:,.0f}", delta_color="off")',
        'm3.metric(f"Upper Bound ({int(conf_level*100)}% CI)", f"${high_b:,.0f}", delta=f"+${high_b - est:,.0f}", delta_color="off")',
        '',
        'range_df = pd.DataFrame({',
        '    "Tier": ["Conservative (Lower CI)", "Point Estimate", "Optimistic (Upper CI)"],',
        '    "Valuation ($ USD)": [low_b, est, high_b],',
        '})',
        'st.bar_chart(range_df.set_index("Tier"))',
    ]
    st_path = os.path.join(output_dir, "streamlit_app.py")
    with open(st_path, "w", encoding="utf-8") as f:
        f.write("\n".join(st_lines) + "\n")
    created["streamlit_app.py"] = st_path

    reqs = "\n".join([
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.7.0",
        "streamlit>=1.36.0",
        "joblib>=1.4.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "lightgbm>=4.3.0",
    ]) + "\n"
    req_path = os.path.join(output_dir, "requirements_deploy.txt")
    with open(req_path, "w", encoding="utf-8") as f:
        f.write(reqs)
    created["requirements_deploy.txt"] = req_path

    docker_lines = [
        "FROM python:3.10-slim",
        "",
        "WORKDIR /app",
        "",
        "RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*",
        "",
        f"COPY {output_dir}/requirements_deploy.txt ./requirements_deploy.txt",
        "RUN pip install --no-cache-dir -r requirements_deploy.txt",
        "",
        "COPY src/ ./src/",
        "COPY models/ ./models/",
        f"COPY {output_dir}/ ./{output_dir}/",
        "",
        "EXPOSE 8000 8501",
        "",
        f'CMD ["uvicorn", "{output_dir}.api:app", "--host", "0.0.0.0", "--port", "8000"]',
    ]
    docker_path = os.path.join(output_dir, "Dockerfile")
    with open(docker_path, "w", encoding="utf-8") as f:
        f.write("\n".join(docker_lines) + "\n")
    created["Dockerfile"] = docker_path

    render_lines = [
        "services:",
        "  - type: web",
        "    name: ames-housing-valuation-api",
        "    runtime: python",
        "    plan: free",
        "    buildCommand: pip install -r app/requirements_deploy.txt",
        f"    startCommand: uvicorn {output_dir}.api:app --host 0.0.0.0 --port $PORT",
        "    envVars:",
        "      - key: MODEL_BUNDLE_PATH",
        "        value: models/ames_housing_bundle.joblib",
        "      - key: PYTHONPATH",
        "        value: .",
        "",
        "  - type: web",
        "    name: ames-housing-valuation-ui",
        "    runtime: python",
        "    plan: free",
        "    buildCommand: pip install -r app/requirements_deploy.txt",
        f"    startCommand: streamlit run {output_dir}/streamlit_app.py --server.port $PORT --server.address 0.0.0.0",
        "    envVars:",
        "      - key: MODEL_BUNDLE_PATH",
        "        value: models/ames_housing_bundle.joblib",
        "      - key: PYTHONPATH",
        "        value: .",
    ]
    render_path = os.path.join(output_dir, "render.yaml")
    with open(render_path, "w", encoding="utf-8") as f:
        f.write("\n".join(render_lines) + "\n")
    created["render.yaml"] = render_path

    return created


def run_packaging_pipeline(
    data_dir: str = os.path.join("data", "processed"),
    artifact_dir: str = os.path.join("models"),
    output_dir: str = "app",
) -> Dict[str, Any]:
    """
    End-to-end execution helper: loads selected features from `data/processed`,
    builds the serialized model bundle in `models/`, and writes `api.py`,
    `streamlit_app.py`, `Dockerfile`, `requirements_deploy.txt`, and `render.yaml`
    into `output_dir` (default: `app`).
    """
    x_path = os.path.join(data_dir, "X_train_selected.csv")
    y_path = os.path.join(data_dir, "y_train_log.csv")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Missing {x_path} or {y_path}. Run Module 4 first to export processed datasets."
        )

    X_train = pd.read_csv(x_path)
    y_train_log = pd.read_csv(y_path).squeeze()

    meta = build_and_save_model_bundle(
        X_train=X_train,
        y_train_log=y_train_log,
        artifact_dir=artifact_dir,
    )
    files = write_deployment_files(
        output_dir=output_dir,
        bundle_rel_path=os.path.join(artifact_dir, "ames_housing_bundle.joblib").replace("\\", "/"),
    )
    return {"metadata": meta, "generated_files": files}


if __name__ == "__main__":
    summary = run_packaging_pipeline()
    print("Model Bundle Metadata:")
    print(json.dumps(summary["metadata"], indent=2))
    print("\nGenerated Deployment Files:")
    for k, v in summary["generated_files"].items():
        print(f"  - {k}: {v}")