# Real Estate Price Prediction (`real-estate-price-prediction`)

An end-to-end Business Analytics and Machine Learning application that estimates residential property market values with prediction confidence intervals. Spanning synthetic data augmentation, exploratory analysis, feature engineering, regression modeling, cloud deployment, and production drift monitoring.

---

## 1. Business Problem & Objective

Real estate listing platforms, sellers, and buyers need a fast, defensible estimate of a property's fair market value prior to a formal appraisal. Overpricing deters qualified buyers and increases time-on-market, while underpricing leaves equity on the table.

**Core Business Question:**

> *"Given a property's structural characteristics, neighborhood context, and macroeconomic conditions, what is its estimated fair market price, and how confident is that estimate?"*

---

## 2. Repository Structure

```text
real-estate-price-prediction/
├── app/
│   ├── api.py                   # FastAPI inference service returning price + confidence interval
│   ├── streamlit_app.py         # Interactive Streamlit valuation dashboard for agents & buyers
│   ├── Dockerfile               # Container definition for cloud deployment
│   ├── render.yaml              # Render Blueprint configuration (API + UI services)
│   └── requirements_deploy.txt  # Lightweight production runtime dependencies
├── data/
│   ├── train.csv                # Base Kaggle Ames Housing training dataset (n = 1,460)
│   ├── test.csv                 # Base Kaggle Ames Housing test dataset (n = 1,459)
│   ├── processed/               # Augmented, cleaned, and model-ready feature matrices
│   └── engineered/              # Feature rankings, VIF diagnostics, and audit reports
├── figures/                     # EDA, cleaning, and multicollinearity visualizations
├── models/
│   ├── ames_housing_bundle.joblib # Serialized LightGBM model, preprocessor & residual quantiles
│   └── model_metadata.json      # Model version, baseline metrics & feature schema
├── notebooks/
│   ├── 1_data_generation.ipynb  # Synthetic contextual feature generation
│   ├── 2_data_analysis.ipynb    # Exploratory Data Analysis (EDA) & price drivers
│   ├── 3_data_cleaning.ipynb    # Missing value imputation, outlier treatment & standardization
│   ├── 4_Feature_Engineering.ipynb # Derived features, encoding, scaling & VIF selection
│   ├── 5_Modeling_Evaluation.ipynb # Model benchmarking, k-fold CV & residual analysis
│   └── 7_Model_Monitoring.ipynb # Data drift detection, rolling error tracking & retraining triggers
├── reports/
│   ├── 1_summary_...md – 7_summary_...md # Stage-by-stage technical & business summaries
│   ├── module7_monitoring_dashboard.html # Standalone HTML governance & drift dashboard
│   ├── module7_feature_drift_report.csv  # PSI, KS-test & Wasserstein drift metrics
│   ├── module7_rolling_error_metrics.csv # Batch & rolling window RMSE/MAE tracking
│   └── figures/                 # Drift and rolling error charts
├── src/
│   ├── data_generation.py       # Reproducible synthetic feature simulation
│   ├── eda_analysis.py          # Statistical summaries & plotting helpers
│   ├── data_cleaning.py         # Preprocessing & data quality pipeline
│   ├── feature_engineering.py   # Feature creation, transformation & selection
│   ├── modeling.py              # Model training, cross-validation & evaluation
│   ├── deployment.py            # Model bundling & inference helpers
│   └── monitoring.py            # Drift detection, HTML report & retraining trigger logic
├── data_dictionary.md           # Documentation of all synthetic contextual features
├── requirements.txt             # Full development & modeling dependencies
└── README.md                    # Project overview and usage guide
```

---

## 3. End-to-End Pipeline Overview

| Module                             | Focus Area                              | Key Highlights                                                                                                                                                                                                                                        |
| :--------------------------------- | :-------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Data Generation**       | Business Understanding & Synthetic Data | Enriches the Kaggle Ames Housing dataset with **13 synthetic contextual features** (amenity distances, renovation history, days-on-market, mortgage rates, local price index, unemployment rate). Documented in `data_dictionary.md`.         |
| **2. EDA**                   | Exploratory Data Analysis               | Analyzes `SalePrice` skewness, neighborhood price tiers, structural correlations, and bivariate relationships (`figures/`).                                                                                                                       |
| **3. Data Cleaning**         | Preprocessing & Outlier Handling        | Resolves structural vs. random missing values, standardizes categorical labels, and treats extreme outliers with documented before/after comparisons.                                                                                                 |
| **4. Feature Engineering**   | Transformation & Selection              | Constructs domain ratios (living area, property age, composite amenity score), applies `log1p` target transformation, checks multicollinearity (VIF), and selects the top **60 predictive features**.                                         |
| **5. Modeling & Evaluation** | Regression Benchmarking                 | Compares Linear/Ridge/Lasso, Random Forest, XGBoost, and **LightGBM** (`LGBMRegressor`) via 5-fold cross-validation (`RMSE`, `MAE`, `MAPE`, $R^2$) and residual diagnostics (Baseline 5-Fold CV `log1p` RMSE: `0.13207`).         |
| **6. Cloud Deployment**      | FastAPI & Streamlit on Render           | Packages the trained model bundle (`ames_housing_bundle.joblib`) into a REST API (`app/api.py`) and interactive Streamlit web app (`app/streamlit_app.py`), deployed via `app/render.yaml`.                                                   |
| **7. Model Monitoring**      | Drift & Retraining Governance           | Simulates weekly production batches, monitors feature drift (PSI, KS-test, Wasserstein distance) and rolling prediction error (`RMSE`/`MAE`), generates an HTML dashboard, and triggers automated retraining alerts when thresholds are breached. |

---

## 4. Quickstart & Local Setup

### Prerequisites

* Python 3.10+
* `pip` or `conda`

### Installation

```bash
git clone https://github.com/jonathansixtCode/real-estate-price-prediction.git
cd real-estate-price-prediction

python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Pipeline

Run the notebooks sequentially inside `notebooks/` (`1_data_generation.ipynb` through `7_Model_Monitoring.ipynb`) from the repository root, or import the modular Python pipeline directly from `src/`.

---

## 5. Running the Valuation App Locally

### 1. Start the FastAPI Backend

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

* Interactive API documentation (Swagger UI): `http://localhost:8000/docs`

### 2. Start the Streamlit Valuation Interface

```bash
streamlit run app/streamlit_app.py
```

* Web interface: `http://localhost:8501`

---

## 6. Cloud Deployment (Render)

The application is configured for multi-service deployment on **Render** via `app/render.yaml`:

* **API Service (`ames-housing-valuation-api`):** Serves real-time price predictions and confidence intervals over HTTP.
* **UI Service (`ames-housing-valuation-ui`):** Provides an agent/buyer-facing property valuation interface connected to the prediction API.

---

## 7. License & Academic Context

Developed as an end-to-end Business Analytics group project demonstrating the full lifecycle of a machine learning valuation product—from raw data augmentation to cloud deployment and post-deployment monitoring.
