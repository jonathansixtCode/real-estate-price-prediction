# Module 6: Model Packaging, REST API & Cloud Deployment

**Project:** Ames Housing Price Prediction (`real-estate-price-prediction`)

**Phase:** Module 6 – Production Packaging, Inference Interface & Cloud Hosting

**Key Artifacts:** `src/deployment.py`, `models/ames_housing_bundle.joblib`, `models/model_metadata.json`, `app/api.py`, `app/streamlit_app.py`, `app/Dockerfile`, `app/render.yaml`, `app/requirements_deploy.txt`

## 1. Overview & Objectives

In Module 6, the top-performing regression model evaluated in Module 5 was packaged into a reproducible, production-ready inference pipeline and deployed to the cloud. The primary objective was to move beyond standalone scripts and expose the model through standardized interfaces both as a REST API for automated systems and as an interactive web application for end users such as real estate agents and prospective home buyers.

## 2. Model Packaging & Inference Pipeline (`src/deployment.py`)

To guarantee consistent predictions without feature drift between training and production, all required artifacts were serialized into a self-contained model bundle (`models/ames_housing_bundle.joblib`).

### 2.1 Production Model Specifications

* **Model Architecture:** `LGBMRegressor` (LightGBM)
* **Number of Features:** $60$ selected and scaled features from Module 4 (`X_train_selected.csv`)
* **Validation Performance (5-Fold CV RMSE on Log Scale):** $\text{RMSE}_{\log} = 0.13207$
* **Out-of-Fold Residual Standard Deviation (Log Scale):** $\sigma_{\log} = 0.13210$
* **Empirical 95% Residual Interval (Log Scale):** $[-0.30704,\; 0.24766]$

### 2.2 Calibrated Confidence & Prediction Intervals

Because real estate valuations in practice require quantifying uncertainty alongside a point estimate, the `predict_with_confidence()` function computes the following for each request:

1. The log-scale point prediction $\hat{y}_{\log}$ and its back-transformation into US dollars via $\hat{y}_{\text{USD}} = \exp(\hat{y}_{\log}) - 1$.
2. A calibrated prediction interval derived from the out-of-fold residual standard deviation $\sigma_{\log}$ and the requested confidence level ($1 - \alpha \in \{0.80, 0.90, 0.95, 0.99\}$):

$$
\text{Lower Bound}_{\text{USD}} = \exp(\hat{y}_{\log} - z \cdot \sigma_{\log}) - 1, \quad \text{Upper Bound}_{\text{USD}} = \exp(\hat{y}_{\log} + z \cdot \sigma_{\log}) - 1
$$

3. Automatic imputation of omitted input features using the training medians (`feature_medians`) stored inside the bundle, ensuring partial feature payloads are handled robustly.

## 3. Project Structure & Deployment Artifacts (`app/`)

All files generated for production inference and cloud deployment were organized cleanly inside the `app/` directory:

| File                            | Purpose & Functionality                                                                                                                                                                |
| :------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/api.py`                  | **FastAPI REST Service** exposing `/health`, `/features`, and `/predict`, including an automatic redirect from `/` to `/docs`.                                         |
| `app/streamlit_app.py`        | **Interactive Streamlit Web App** (*Ames Property Valuation Studio*) featuring sliders for key value drivers, selectable confidence levels, and visual valuation range charts. |
| `app/requirements_deploy.txt` | Lightweight runtime dependency manifest (`fastapi`, `uvicorn`, `pydantic`, `streamlit`, `joblib`, `numpy`, `pandas`, `scikit-learn`, `lightgbm`).                    |
| `app/Dockerfile`              | Container definition based on `python:3.10-slim` (including `libgomp1` for LightGBM) for portable containerized deployment.                                                        |
| `app/render.yaml`             | Infrastructure-as-Code Blueprint defining the multi-service cloud deployment on Render.                                                                                                |

## 4. Cloud Deployment & Live Testing Links

Both services were deployed via the Render Blueprint (`app/render.yaml`) directly from the GitHub repository (`jonathansixtCode/real-estate-price-prediction`, `main` branch) and verified in production.

### 4.1 Live URLs

* **Interactive User Interface (Streamlit UI):**

  [https://ames-housing-valuation-ui.onrender.com](https://ames-housing-valuation-ui.onrender.com)
* **REST API & Interactive Swagger Documentation (FastAPI):**

  [https://ames-housing-valuation-api.onrender.com](https://ames-housing-valuation-api.onrender.com) *(automatically redirects to `/docs`)*
* **API Health-Check Endpoint:**

  [https://ames-housing-valuation-api.onrender.com/health](https://ames-housing-valuation-api.onrender.com/health)
* **API Feature Schema Endpoint:**

  [https://ames-housing-valuation-api.onrender.com/features](https://ames-housing-valuation-api.onrender.com/features)

*(Note regarding the Cloud Free Tier: After periods of inactivity, free web instances spin down into sleep mode. The initial request after an idle period may therefore take approximately 30–60 seconds for a cold start.)*

### 4.2 Verified API Test (`POST /predict`)

The live API was verified through the interactive Swagger UI (`/docs`):

* **Sample Request (`POST /predict`):**

  ```json
  {
    "features": {
      "OverallQual": 1.0,
      "TotalSF": 0.8
    },
    "confidence_level": 0.95
  }
  ```
* **Live Response (`HTTP 200 OK`):**

  ```json
  {
    "estimated_price_usd": 180229.45,
    "confidence_level": 0.95,
    "confidence_range_usd": {
      "lower_bound": 139116.96,
      "upper_bound": 233491.6
    },
    "log_prediction": 12.10199,
    "model_type": "LGBMRegressor"
  }
  ```

## 5. Key Takeaways from Module 6

1. **Self-Contained Model Bundles Prevent Skew:** Storing the trained estimator, exact feature ordering, training medians, min/max bounds, and residual statistics in a single `.joblib` bundle decouples inference from the training pipeline.
2. **Path Resilience Across Local & Cloud Environments:** Dynamic path resolution (`resolve_bundle_path()`) combined with environment variable configuration (`MODEL_BUNDLE_PATH`) allows both `uvicorn app.api:app` and `streamlit run app/streamlit_app.py` to run seamlessly on Windows locally and inside Linux cloud containers without code modifications.
3. **Decoupling API and UI Improves Maintainability:** Hosting the FastAPI backend and Streamlit frontend as two independent web services enables separate testing, monitoring, and scaling for programmatic consumers and human end users.
