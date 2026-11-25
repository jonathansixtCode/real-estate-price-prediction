# Module 5 Summary: Regression Modeling, Cross-Validation & Residual Analysis

## 1. Overview & Objectives

The primary objective of **Module 5** is to train, benchmark, and diagnose multiple regression architectures on the leak-free, 60-feature matrix exported from Module 4 (`X_train_selected.csv`: $1,458 \times 60$; `X_test_selected.csv`: $1,459 \times 60$).

* **Validation Strategy:** 5-Fold Cross-Validation (`shuffle=True`, `random_state=42`) evaluated across both the `log1p`-transformed target scale (`RMSE_log`, $R^2_{\log}$) and back-transformed US Dollar scale (`RMSE_USD`, `MAE_USD`, `MAPE %`, $R^2_{\text{USD}}$).
* **Candidate Models Evaluated:**
  * Linear Baselines: `Linear Regression`, `Ridge Regression (alpha=10)`, `Lasso Regression (alpha=0.001)`
  * Tree-Based Ensembles: `Random Forest`, `XGBoost Regressor`, `LightGBM Regressor`

---

## 2. 5-Fold Cross-Validation Benchmark Results

Across all six candidate regressors, **LightGBM Regressor** achieved the best overall cross-validation performance on the log scale (`RMSE_log = 0.1167`, $R^2_{\log} = 0.9143$), closely followed by **XGBoost Regressor** (`RMSE_log = 0.1171`) and regularized linear models (**Ridge** and **Lasso**).

| Rank | Model | $\text{RMSE}_{\log}$ (Mean $\pm$ Std) | $R^2_{\log}$ | $\text{RMSE}_{\text{USD}}$ | $\text{MAE}_{\text{USD}}$ | $\text{MAPE}$ (%) | $R^2_{\text{USD}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **LightGBM Regressor** | **0.1167 $\pm$ 0.0110** | **0.9143** | **$22,429.74** | **$14,295.07** | **8.19%** | **0.9182** |
| **2** | **XGBoost Regressor** | 0.1171 $\pm$ 0.0124 | 0.9136 | $22,499.78 | **$14,174.99** | **8.14%** | 0.9175 |
| **3** | **Ridge Regression ($\alpha=10$)** | 0.1235 $\pm$ 0.0159 | 0.9038 | $23,316.05 | $15,248.62 | 8.85% | 0.9114 |
| **4** | **Lasso Regression ($\alpha=0.001$)** | 0.1236 $\pm$ 0.0161 | 0.9037 | $23,440.91 | $15,283.39 | 8.87% | 0.9104 |
| **5** | **Linear Regression (OLS)** | 0.1239 $\pm$ 0.0162 | 0.9032 | $23,544.52 | $15,342.19 | 8.91% | 0.9096 |
| **6** | **Random Forest** | 0.1304 $\pm$ 0.0099 | 0.8933 | $25,627.14 | $16,232.78 | 9.33% | 0.8933 |

### Key Modeling Takeaways
* **Gradient Boosting Superiority:** `LightGBM` and `XGBoost` outperform linear models by ~0.0068 log-RMSE (~$\$1,000$ in MAE and ~0.7% in MAPE), capturing non-linear interactions across living area, neighborhood price baselines, and structural quality.
* **Stability Across Folds:** `LightGBM` exhibits lower cross-fold variance ($\text{Std} = 0.0110$) than `XGBoost` ($\text{Std} = 0.0124$) and linear models ($\text{Std} \approx 0.0160$).
* **Strong Linear Baseline:** Due to the engineered interaction terms (`Qual_x_TotalSF`, `Neighborhood_Median_PricePerSqM`) and `log1p` skewness corrections from Module 4, regularized linear models still explain $>90.3\%$ of target variance ($R^2_{\text{USD}} = 0.9114$).

---

## 3. Residual Diagnostics & Assumption Verification (`LightGBM Regressor`)

Out-of-fold residuals ($e_i = y_i - \hat{y}_i$) for the top-performing **LightGBM Regressor** were analyzed to check classical regression assumptions and detect systematic pricing biases:

### A. Global Error & Distribution Statistics
* **Mean Log Residual:** $-0.0005$ ($\text{Std} = 0.1169$) — confirms virtually zero global bias on the log scale.
* **Mean / Median Dollar Residual:** $+\$1,245.57$ / $+\$134.81$ — minor positive skew in dollar terms caused by back-transformation (`expm1`) of extreme high-value luxury properties.
* **Distribution Shape:**
  * **Skewness:** $-0.4493$ (slight left tail from occasional over-predictions on distressed/atypical sales).
  * **Kurtosis:** $4.5777$ (leptokurtic / heavier tails than a pure Gaussian distribution).
  * **Shapiro-Wilk Test:** $W = 0.9467$, $p < 0.0001$ (rejects strict normality due to 25 extreme outliers beyond $\pm 3\text{ SD}$ out of 1,458 observations).

### B. Systematic Under/Over-Prediction by Price Tier
Slicing out-of-fold dollar residuals across actual sale price quartiles reveals mild regression-to-the-mean behavior at the extremes:

| Price Segment | Mean Dollar Residual ($\text{Actual} - \text{Predicted}$) | Interpretation |
| :--- | :---: | :--- |
| **Q1: Budget ($<\$130\text{k}$)** | $-\$2,167.21$ | Slight **over-prediction** on low-end / distressed properties |
| **Q2: Mid-Low** | $-\$405.81$ | Near-zero bias (well-calibrated) |
| **Q3: Mid-High** | $+\$1,393.02$ | Near-zero bias (well-calibrated) |
| **Q4: Luxury ($>\$214\text{k}$)** | $+\$6,177.26$ | Systematic **under-prediction** on high-end luxury estates |

---

## 4. Outlier Audit & Key Predictive Drivers

### A. Largest Percentage Error Outliers
Inspecting the top 10 largest absolute percentage errors highlights that tail errors are primarily driven by atypical transactions rather than systemic model failure:
* **Distressed / Partial / Non-Normal Sales:** Properties sold far below structural value (e.g., actual $\$34,900$ vs. predicted $\$97,679$ [$+179.88\%$ error] in `IDORRail`; actual $\$39,300$ vs. predicted $\$88,690$ [$+125.67\%$] in `BrkSide`).
* **Ultra-Luxury Outliers:** High-end estates in `NoRidge` (e.g., actual $\$755,000$ vs. predicted $\$573,281$ [$-24.07\%$]) where limited training density caps tree leaf predictions.

### B. Top Predictive Features
Feature importance extraction on the fitted `LightGBM Regressor` confirms that engineered domain features from Module 4 dominate split decisions, led by:
1. `Neighborhood_Median_PricePerSqM` & `TotalSF` / `TotalSF_SqM`
2. `Qual_x_TotalSF` & `Qual_x_GrLivArea`
3. `OverallQual`, `PropertyAge`, `YearsSinceRemodel`, and `TotalBathrooms`

---

## 5. Outputs & Next Steps

* **Generated Artifacts:**
  * Model benchmark comparison chart: `../reports/figures/module5_model_comparison.png`
  * 4-panel residual diagnostic dashboard: `../reports/figures/module5_residual_diagnostics.png`
  * Test set predictions ($1,459$ rows): `../submissions/module5_predictions.csv`
* **Recommended Next Steps (Module 6 — Hyperparameter Tuning & Ensembling):**
  * Perform Bayesian/grid hyperparameter optimization for **LightGBM** and **XGBoost** (e.g., tuning `num_leaves`, `max_depth`, `min_child_samples`, and `colsample_bytree`).
  * Build a weighted **Stacking / Blending Ensemble** combining `LightGBM` + `XGBoost` + `Ridge Regression` to reduce Q4 luxury under-prediction and push CV $\text{RMSE}_{\log}$ below `0.1140`.