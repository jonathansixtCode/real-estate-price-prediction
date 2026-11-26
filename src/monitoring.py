"""
Module 7: Model Monitoring, Data Drift Detection & Retraining Triggers
Ames Housing Dataset — Production Observability & Governance Pipeline

Provides modular, production-ready utilities for:
- Simulating realistic incoming production listing streams with controlled covariate & concept drift
- Tracking feature distribution / covariate drift (KS test, Wasserstein distance, PSI)
- Monitoring rolling prediction error over time (RMSE/MAE on log and USD scales) as ground-truth sales arrive
- Generating an interactive Evidently HTML monitoring dashboard (with a rich self-contained HTML fallback)
- Evaluating automated retraining triggers based on drift share, rolling RMSE degradation, and sample volume
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import json
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, RegressionPreset
    HAS_EVIDENTLY = True
except ImportError:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset, RegressionPreset
        HAS_EVIDENTLY = True
    except ImportError:
        HAS_EVIDENTLY = False


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between a reference distribution and a current sample.
    Standard interpretation:
      - PSI < 0.10: No significant shift
      - 0.10 <= PSI < 0.25: Moderate shift (warning)
      - PSI >= 0.25: Severe distribution shift (retraining candidate)
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, buckets + 1)
    bin_edges = np.unique(np.percentile(expected, quantiles))
    if len(bin_edges) < 3:
        bin_edges = np.linspace(min(expected.min(), actual.min()), max(expected.max(), actual.max()) + 1e-6, buckets + 1)
    else:
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    eps = 1e-6
    exp_pct = np.clip(exp_counts / max(len(expected), 1), eps, 1.0)
    act_pct = np.clip(act_counts / max(len(actual), 1), eps, 1.0)

    psi_val = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi_val, 5)


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    ks_alpha: float = 0.05,
    psi_threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Compare incoming listing feature distributions against the training reference dataset
    using Kolmogorov-Smirnov two-sample tests, Wasserstein distance, and PSI.
    """
    if feature_cols is None:
        feature_cols = [c for c in reference_df.columns if c in current_df.columns]

    records: List[Dict[str, Any]] = []
    for col in feature_cols:
        ref_vals = reference_df[col].dropna().to_numpy(dtype=float)
        cur_vals = current_df[col].dropna().to_numpy(dtype=float)

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)
        w_dist = float(stats.wasserstein_distance(ref_vals, cur_vals))
        psi_val = compute_psi(ref_vals, cur_vals)

        ref_mean, cur_mean = float(np.mean(ref_vals)), float(np.mean(cur_vals))
        ref_std, cur_std = float(np.std(ref_vals)), float(np.std(cur_vals))
        mean_shift_z = (cur_mean - ref_mean) / (ref_std + 1e-8)

        drifted = bool((ks_pval < ks_alpha and psi_val >= 0.10) or (psi_val >= psi_threshold))

        records.append({
            "feature": col,
            "ref_mean": round(ref_mean, 4),
            "cur_mean": round(cur_mean, 4),
            "mean_shift_z": round(float(mean_shift_z), 4),
            "ref_std": round(ref_std, 4),
            "cur_std": round(cur_std, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pval), 6),
            "wasserstein_dist": round(w_dist, 4),
            "psi": psi_val,
            "drift_detected": drifted,
        })

    drift_df = pd.DataFrame(records)
    if not drift_df.empty:
        drift_df = drift_df.sort_values(by=["drift_detected", "psi", "ks_statistic"], ascending=[False, False, False]).reset_index(drop=True)
    return drift_df


def simulate_production_stream(
    X_base: pd.DataFrame,
    bundle: Dict[str, Any],
    n_batches: int = 8,
    batch_size: int = 120,
    drift_start_batch: int = 5,
    start_date: str = "2026-01-05",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simulate weekly batches of incoming property listings and delayed ground-truth sale prices.
    Batches 1..(drift_start_batch-1) reflect stationary market conditions; batches >= drift_start_batch
    introduce progressive market inflation, luxury-segment shift, and mild concept drift.
    """
    rng = np.random.default_rng(random_state)
    feature_names: List[str] = bundle["feature_names"]
    model = bundle["model"]
    std_log: float = float(bundle.get("residual_std_log", 0.132))

    base_dt = datetime.strptime(start_date, "%Y-%m-%d")
    frames: List[pd.DataFrame] = []

    key_shift_features = [
        "OverallQual",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "GarageArea",
        "Neighborhood_Median_PricePerSqM",
        "GarageCars",
    ]

    for b in range(1, n_batches + 1):
        idx = rng.choice(len(X_base), size=batch_size, replace=True)
        batch_X = X_base.iloc[idx][feature_names].copy().reset_index(drop=True)

        # Identify continuous vs. binary one-hot features so binary flags remain valid {0, 1}
        is_binary = {
            col: set(X_base[col].dropna().unique()).issubset({0.0, 1.0, 0, 1})
            for col in feature_names
        }
        for col in feature_names:
            if not is_binary[col]:
                batch_X[col] = batch_X[col] + rng.normal(0.0, 0.05, size=batch_size)

        # Baseline model prediction on incoming features
        pred_log = model.predict(batch_X)

        # Simulate true underlying log-price generation
        concept_shift_log = 0.0
        if b >= drift_start_batch:
            severity = (b - drift_start_batch + 1) / (n_batches - drift_start_batch + 1)
            for feat in key_shift_features:
                if feat in batch_X.columns:
                    direction = 1.0
                    batch_X[feat] = batch_X[feat] + direction * (0.45 * severity)
            # Recompute predictions after covariate shift
            pred_log = model.predict(batch_X)
            # Introduce macro market price inflation (concept drift) not yet seen by static model
            concept_shift_log = 0.08 * severity

        true_log = pred_log + concept_shift_log + rng.normal(0.0, std_log, size=batch_size)

        batch_date = base_dt + timedelta(weeks=b - 1)
        batch_X["batch_id"] = b
        batch_X["batch_date"] = batch_date.strftime("%Y-%m-%d")
        batch_X["prediction_log"] = pred_log
        batch_X["target_log"] = true_log
        batch_X["prediction_usd"] = np.expm1(pred_log)
        batch_X["target_usd"] = np.expm1(true_log)
        frames.append(batch_X)

    return pd.concat(frames, axis=0, ignore_index=True)


def compute_rolling_error_metrics(
    stream_df: pd.DataFrame,
    window_size: int = 150,
    step_size: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute both batch-level and sliding-window rolling error metrics (RMSE, MAE, MAPE, R2)
    on log1p and USD scales as actual closing prices arrive.
    """
    batch_rows: List[Dict[str, Any]] = []
    for (b_id, b_date), grp in stream_df.groupby(["batch_id", "batch_date"], sort=True):
        y_true_log = grp["target_log"].to_numpy()
        y_pred_log = grp["prediction_log"].to_numpy()
        y_true_usd = grp["target_usd"].to_numpy()
        y_pred_usd = grp["prediction_usd"].to_numpy()

        rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
        mae_log = float(mean_absolute_error(y_true_log, y_pred_log))
        rmse_usd = float(np.sqrt(mean_squared_error(y_true_usd, y_pred_usd)))
        mae_usd = float(mean_absolute_error(y_true_usd, y_pred_usd))
        mape_usd = float(np.mean(np.abs((y_true_usd - y_pred_usd) / np.clip(y_true_usd, 1.0, None))) * 100.0)
        r2_log = float(r2_score(y_true_log, y_pred_log))

        batch_rows.append({
            "batch_id": int(b_id),
            "batch_date": b_date,
            "n_samples": int(len(grp)),
            "rmse_log": round(rmse_log, 5),
            "mae_log": round(mae_log, 5),
            "rmse_usd": round(rmse_usd, 2),
            "mae_usd": round(mae_usd, 2),
            "mape_pct": round(mape_usd, 2),
            "r2_log": round(r2_log, 4),
        })

    batch_metrics_df = pd.DataFrame(batch_rows)

    rolling_rows: List[Dict[str, Any]] = []
    n_total = len(stream_df)
    if n_total >= window_size:
        for end_idx in range(window_size, n_total + 1, step_size):
            start_idx = end_idx - window_size
            win = stream_df.iloc[start_idx:end_idx]
            y_true_log = win["target_log"].to_numpy()
            y_pred_log = win["prediction_log"].to_numpy()
            y_true_usd = win["target_usd"].to_numpy()
            y_pred_usd = win["prediction_usd"].to_numpy()

            rolling_rows.append({
                "window_end_idx": int(end_idx),
                "window_end_date": str(win["batch_date"].iloc[-1]),
                "window_size": int(window_size),
                "rolling_rmse_log": round(float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))), 5),
                "rolling_mae_log": round(float(mean_absolute_error(y_true_log, y_pred_log)), 5),
                "rolling_rmse_usd": round(float(np.sqrt(mean_squared_error(y_true_usd, y_pred_usd))), 2),
                "rolling_mae_usd": round(float(mean_absolute_error(y_true_usd, y_pred_usd)), 2),
            })

    rolling_metrics_df = pd.DataFrame(rolling_rows)
    return batch_metrics_df, rolling_metrics_df


def evaluate_retraining_triggers(
    drift_df: pd.DataFrame,
    batch_metrics_df: pd.DataFrame,
    baseline_rmse_log: float = 0.13207,
    max_drifted_feature_share: float = 0.15,
    max_psi_threshold: float = 0.25,
    rmse_degradation_factor: float = 1.20,
    min_new_labeled_sales: int = 250,
) -> Dict[str, Any]:
    """
    Evaluate production governance rules to decide whether automated model retraining
    should be triggered.
    """
    n_features = max(len(drift_df), 1)
    n_drifted = int(drift_df["drift_detected"].sum()) if not drift_df.empty else 0
    drifted_share = round(n_drifted / n_features, 4)
    max_psi = float(drift_df["psi"].max()) if not drift_df.empty else 0.0

    latest_rmse_log = float(batch_metrics_df["rmse_log"].iloc[-1]) if not batch_metrics_df.empty else baseline_rmse_log
    rmse_threshold = round(baseline_rmse_log * rmse_degradation_factor, 5)
    total_new_sales = int(batch_metrics_df["n_samples"].sum()) if not batch_metrics_df.empty else 0

    triggers = {
        "covariate_drift_share_exceeded": {
            "triggered": bool(drifted_share >= max_drifted_feature_share),
            "observed": drifted_share,
            "threshold": max_drifted_feature_share,
            "description": f"Share of drifted features ({drifted_share:.1%}) >= {max_drifted_feature_share:.1%}",
        },
        "severe_feature_psi_exceeded": {
            "triggered": bool(max_psi >= max_psi_threshold),
            "observed": round(max_psi, 5),
            "threshold": max_psi_threshold,
            "description": f"Maximum feature PSI ({max_psi:.4f}) >= {max_psi_threshold}",
        },
        "rolling_rmse_degradation_exceeded": {
            "triggered": bool(latest_rmse_log >= rmse_threshold),
            "observed": round(latest_rmse_log, 5),
            "threshold": rmse_threshold,
            "description": f"Latest batch RMSE_log ({latest_rmse_log:.5f}) >= {rmse_threshold:.5f} (+20% over baseline {baseline_rmse_log:.5f})",
        },
        "new_labeled_volume_ready": {
            "triggered": bool(total_new_sales >= min_new_labeled_sales),
            "observed": total_new_sales,
            "threshold": min_new_labeled_sales,
            "description": f"Accumulated closed sales ({total_new_sales}) >= {min_new_labeled_sales} minimum retraining sample size",
        },
    }

    active_alerts = [k for k, v in triggers.items() if v["triggered"] and k != "new_labeled_volume_ready"]
    retrain_recommended = bool(len(active_alerts) > 0 and triggers["new_labeled_volume_ready"]["triggered"])

    return {
        "retraining_recommended": retrain_recommended,
        "active_alert_count": len(active_alerts),
        "active_alerts": active_alerts,
        "triggers": triggers,
        "recommended_action": (
            "TRIGGER RETRAINING PIPELINE: Re-run Modules 3–6 on combined historical + recent closed sales and validate challenger model before promotion."
            if retrain_recommended
            else "NO RETRAINING NEEDED: Continue scheduled monitoring."
        ),
    }


def generate_monitoring_dashboard(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    batch_metrics_df: pd.DataFrame,
    trigger_decision: Dict[str, Any],
    output_html_path: str = os.path.join("reports", "monitoring_dashboard.html"),
) -> Dict[str, Any]:
    """
    Generate an interactive HTML monitoring dashboard using Evidently (when installed)
    or a clean, self-contained HTML governance report as fallback.
    """
    os.makedirs(os.path.dirname(output_html_path) or ".", exist_ok=True)
    feature_cols = [c for c in drift_df["feature"].tolist() if c in reference_df.columns and c in current_df.columns]

    if HAS_EVIDENTLY:
        try:
            ref_eval = reference_df[feature_cols].copy()
            cur_eval = current_df[feature_cols].copy()
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_eval, current_data=cur_eval)
            report.save_html(output_html_path)
            return {
                "dashboard_path": output_html_path,
                "engine": "Evidently",
                "status": "success",
            }
        except Exception:
            pass

    top_drift_html = drift_df.head(15).to_html(index=False, classes="tbl", border=0)
    batch_html = batch_metrics_df.to_html(index=False, classes="tbl", border=0)
    status_badge = "RETRAINING TRIGGERED" if trigger_decision["retraining_recommended"] else "HEALTHY / MONITORING"
    badge_color = "#b91c1c" if trigger_decision["retraining_recommended"] else "#15803d"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Ames Housing Valuation — Model Monitoring Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 32px; background: #f8fafc; color: #0f172a; }}
    .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #cbd5e1; padding-bottom: 16px; margin-bottom: 24px; }}
    .badge {{ background: {badge_color}; color: #fff; padding: 8px 14px; border-radius: 6px; font-weight: 600; font-size: 14px; }}
    .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-bottom: 24px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
    table.tbl {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    table.tbl th, table.tbl td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
    table.tbl th {{ background: #f1f5f9; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1 style="margin:0;">Ames Real Estate Valuation — Production Monitoring Dashboard</h1>
      <p style="margin:4px 0 0; color:#475569;">Covariate Drift, Rolling Error Tracking & Automated Retraining Governance</p>
    </div>
    <span class="badge">{status_badge}</span>
  </div>
  <div class="card">
    <h2>1. Governance & Retraining Decision</h2>
    <p><strong>Recommended Action:</strong> {trigger_decision["recommended_action"]}</p>
    <p><strong>Active Alerts ({trigger_decision["active_alert_count"]}):</strong> {", ".join(trigger_decision["active_alerts"]) or "None"}</p>
  </div>
  <div class="card">
    <h2>2. Batch Performance Over Time (Log & USD Scales)</h2>
    {batch_html}
  </div>
  <div class="card">
    <h2>3. Top 15 Features Ranked by Drift (PSI & KS Statistic)</h2>
    {top_drift_html}
  </div>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return {
        "dashboard_path": output_html_path,
        "engine": "Built-in Governance HTML Dashboard",
        "status": "success",
    }


def run_monitoring_pipeline(
    data_dir: str = os.path.join("data", "processed"),
    artifact_dir: str = os.path.join("models"),
    bundle_filename: str = "ames_housing_bundle.joblib",
    report_dir: str = os.path.join("reports"),
) -> Dict[str, Any]:
    """
    End-to-end execution helper for Module 7:
    1. Loads reference training features and the production LightGBM bundle
    2. Simulates 8 weekly batches of incoming listings & realized closing prices
    3. Computes feature drift (KS, Wasserstein, PSI) comparing recent batches vs. training reference
    4. Tracks batch & rolling RMSE/MAE over time
    5. Evaluates retraining triggers and exports dashboard & JSON audit summary
    """
    x_train_path = os.path.join(data_dir, "X_train_selected.csv")
    x_test_path = os.path.join(data_dir, "X_test_selected.csv")
    bundle_path = os.path.join(artifact_dir, bundle_filename)

    if not os.path.exists(x_train_path) or not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Missing {x_train_path} or {bundle_path}. Ensure Modules 4 and 6 have been executed."
        )

    X_train = pd.read_csv(x_train_path)
    X_stream_source = pd.read_csv(x_test_path) if os.path.exists(x_test_path) else X_train
    bundle = joblib.load(bundle_path)
    feature_names: List[str] = bundle["feature_names"]
    baseline_rmse_log = float(bundle.get("cv_rmse_log", 0.13207))

    stream_df = simulate_production_stream(
        X_base=X_stream_source,
        bundle=bundle,
        n_batches=8,
        batch_size=120,
        drift_start_batch=5,
        random_state=42,
    )

    # Compare recent production window (batches 5..8) against training reference
    recent_window_df = stream_df[stream_df["batch_id"] >= 5].copy()
    drift_df = detect_feature_drift(
        reference_df=X_train[feature_names],
        current_df=recent_window_df[feature_names],
        feature_cols=feature_names,
    )

    batch_metrics_df, rolling_metrics_df = compute_rolling_error_metrics(
        stream_df=stream_df,
        window_size=180,
        step_size=60,
    )

    trigger_decision = evaluate_retraining_triggers(
        drift_df=drift_df,
        batch_metrics_df=batch_metrics_df,
        baseline_rmse_log=baseline_rmse_log,
    )

    os.makedirs(report_dir, exist_ok=True)
    drift_csv_path = os.path.join(report_dir, "module7_feature_drift_report.csv")
    batch_csv_path = os.path.join(report_dir, "module7_rolling_error_metrics.csv")
    audit_json_path = os.path.join(report_dir, "module7_monitoring_audit.json")
    dashboard_html_path = os.path.join(report_dir, "module7_monitoring_dashboard.html")

    drift_df.to_csv(drift_csv_path, index=False)
    batch_metrics_df.to_csv(batch_csv_path, index=False)

    dash_info = generate_monitoring_dashboard(
        reference_df=X_train,
        current_df=recent_window_df,
        drift_df=drift_df,
        batch_metrics_df=batch_metrics_df,
        trigger_decision=trigger_decision,
        output_html_path=dashboard_html_path,
    )

    audit_summary = {
        "model_type": bundle.get("model_type", "LGBMRegressor"),
        "baseline_cv_rmse_log": baseline_rmse_log,
        "n_monitored_features": len(feature_names),
        "n_drifted_features": int(drift_df["drift_detected"].sum()),
        "drifted_feature_share": round(float(drift_df["drift_detected"].mean()), 4),
        "top_drifted_features": drift_df.head(5)[["feature", "psi", "ks_statistic", "ks_pvalue"]].to_dict(orient="records"),
        "trigger_decision": trigger_decision,
        "artifacts": {
            "feature_drift_csv": drift_csv_path,
            "batch_metrics_csv": batch_csv_path,
            "monitoring_dashboard_html": dash_info["dashboard_path"],
            "dashboard_engine": dash_info["engine"],
        },
    }

    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    return audit_summary


if __name__ == "__main__":
    summary = run_monitoring_pipeline()
    print(json.dumps(summary, indent=2))