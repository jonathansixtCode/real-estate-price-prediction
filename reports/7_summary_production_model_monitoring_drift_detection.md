# Module 7: Production Model Monitoring, Drift Detection & Automated Retraining

## 1. Overview & Objectives

This module demonstrates end-to-end **Production Model Monitoring** for the Ames Housing valuation model (`LGBMRegressor`, 60 features). The primary goal is to detect gradual shifts in incoming feature distributions (*Data Drift*) as well as model performance degradation (*Concept Drift*) at an early stage, enabling automated retraining decisions.

* **Baseline Model:** `lgbm_ames_model.joblib` (60 selected features)
* **Reference Dataset:** `X_train_selected.csv` ($n = 1,458$)
* **Reference Metric (5-Fold CV RMSE, `log1p`):** `0.13207`
* **Monitoring Engine:** `src/monitoring.py` (automatically falling back to a custom standalone HTML governance report since `Evidently` is not installed in the environment)

## 2. Production Stream Simulation

To simulate realistic production behavior over time, **8 weekly batches** of 120 property sales each (totaling **960 transactions**) are generated:

* **Weeks 1–4 (Stationary Phase):** Incoming data distributions closely match the historical training data.
* **Weeks 5–8 (Market Shift / Drift Phase):** A progressive market shift occurs:
  * **Covariate Drift:** Increasing proportion of newer, larger, and higher-end properties.
  * **Concept Drift:** Rising overall price levels alter the underlying relationship between property features and sale prices.

## 3. Data Drift Analysis (Weeks 5–8 vs. Reference)

Comparing the shifted production window ($n = 480$) against the training reference ($n = 1,458$) using statistical metrics (**Population Stability Index [PSI]**, **Kolmogorov-Smirnov [KS] test**, and **Wasserstein Distance**) yields the following results:

* **Overall Drift Rate:** Statistically significant drift detected in **18 out of 60 features (30.0%)**.
* **Top Drifting Features:**
  1. `IsNewBuild`: PSI = `4.43` (severe drift)
  2. `BsmtExposure`: PSI = `1.25` (major drift)
  3. `BsmtCond`: PSI = `1.22` (major drift)

## 4. Rolling Prediction Error Tracking & Performance Degradation

Prediction accuracy is monitored both per weekly batch and across a rolling window (`window_size = 180`) using **RMSE** and **MAE** (on both `log1p` and dollar scales):

* **Alert Threshold:** `+20%` degradation relative to the baseline RMSE (`0.13207` $\rightarrow$ **`0.15848`**).
* **Error Trajectory:**
  * During **Weeks 1–5**, prediction errors remain stable within expected bounds.
  * Starting in **Batch 6**, prediction errors steadily climb.
  * By **Batch 8**, `rmse_log` reaches **`0.16518`** (`+25.1%` above baseline), breaching the critical alert threshold.

## 5. Governance & Automated Retraining (Concluding Steps)

Using the monitoring utilities imported from `src/monitoring.py`, the workflow concludes with two automated governance steps:

1. **Interactive Monitoring Dashboard (`generate_monitoring_dashboard`):** Generates a standalone HTML report visualizing feature distribution shifts, PSI scores, and rolling error trajectories.
2. **Automated Retraining Triggers (`evaluate_retraining_triggers`):** Evaluates the configured thresholds (30% feature drift rate and >20% RMSE degradation) and automatically triggers a retraining recommendation when breached.