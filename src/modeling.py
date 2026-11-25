"""
Module 5: Regression Modeling, Evaluation & Residual Analysis
Ames Housing Dataset — Predictive Modeling Pipeline

Provides modular, leak-free functions for:
- Initializing candidate linear, regularized, and gradient-boosted regressors
- Performing K-Fold cross-validation with comprehensive metrics (RMSE, MAE, MAPE, R2)
  evaluated both on the log-transformed target and back-transformed dollar scale
- Diagnosing residuals to verify regression assumptions and identify systematic bias
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Initialize baseline and ensemble regression models for comparison.

    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary mapping model names to un-fitted scikit-learn compatible estimators.
    """
    models: Dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression (alpha=10)": Ridge(alpha=10.0, random_state=random_state),
        "Lasso Regression (alpha=0.001)": Lasso(alpha=0.001, max_iter=10000, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=18,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if HAS_XGB:
        models["XGBoost Regressor"] = XGBRegressor(
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

    if HAS_LGBM:
        models["LightGBM Regressor"] = LGBMRegressor(
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

    return models


def compute_regression_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> Dict[str, float]:
    """
    Compute evaluation metrics in both log1p space and original dollar space.

    Parameters
    ----------
    y_true_log : array-like
        Ground truth target values in log1p scale.
    y_pred_log : array-like
        Predicted target values in log1p scale.

    Returns
    -------
    dict
        Dictionary containing RMSE_log, R2_log, RMSE_USD, MAE_USD, MAPE_pct, and R2_USD.
    """
    y_true_log = np.asarray(y_true_log)
    y_pred_log = np.asarray(y_pred_log)

    # Metrics on log1p scale (Kaggle competition standard)
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    r2_log = float(r2_score(y_true_log, y_pred_log))

    # Back-transform to original USD scale using expm1
    y_true_usd = np.expm1(y_true_log)
    # Clip extreme log predictions before expm1 to guard against numerical overflow
    y_pred_usd = np.expm1(np.clip(y_pred_log, a_min=None, a_max=20.0))

    rmse_usd = float(np.sqrt(mean_squared_error(y_true_usd, y_pred_usd)))
    mae_usd = float(mean_absolute_error(y_true_usd, y_pred_usd))
    mape_pct = float(np.mean(np.abs((y_true_usd - y_pred_usd) / y_true_usd)) * 100.0)
    r2_usd = float(r2_score(y_true_usd, y_pred_usd))

    return {
        "RMSE_log": rmse_log,
        "R2_log": r2_log,
        "RMSE_USD": rmse_usd,
        "MAE_USD": mae_usd,
        "MAPE_pct": mape_pct,
        "R2_USD": r2_usd,
    }


def evaluate_models_cv(
    X: pd.DataFrame,
    y_log: pd.Series,
    models: Optional[Dict[str, Any]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Perform K-Fold cross-validation across all candidate regressors.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (training set).
    y_log : pd.Series
        Log1p-transformed target variable (`SalePrice`).
    models : dict, optional
        Dictionary of candidate models. Defaults to `get_candidate_models()`.
    n_splits : int, default=5
        Number of cross-validation folds.
    random_state : int, default=42
        Random seed for KFold splitting.

    Returns
    -------
    summary_df : pd.DataFrame
        Aggregated mean and std of all metrics sorted by RMSE_log ascending.
    oof_predictions : dict
        Out-of-fold log1p predictions for each model.
    fitted_models : dict
        Models fitted on the full training dataset for final inference/inspection.
    """
    if models is None:
        models = get_candidate_models(random_state=random_state)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results_records = []
    oof_predictions: Dict[str, np.ndarray] = {}
    fitted_models: Dict[str, Any] = {}

    for name, estimator in models.items():
        oof_pred = np.zeros(len(X), dtype=float)
        fold_metrics: List[Dict[str, float]] = []

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

            model_clone = clone(estimator)
            model_clone.fit(X_tr, y_tr)
            preds = model_clone.predict(X_val)

            oof_pred[val_idx] = preds
            fold_metrics.append(compute_regression_metrics(y_val, preds))

        oof_predictions[name] = oof_pred
        fold_df = pd.DataFrame(fold_metrics)

        summary_row = {
            "Model": name,
            "RMSE_log_Mean": fold_df["RMSE_log"].mean(),
            "RMSE_log_Std": fold_df["RMSE_log"].std(),
            "R2_log_Mean": fold_df["R2_log"].mean(),
            "RMSE_USD_Mean": fold_df["RMSE_USD"].mean(),
            "MAE_USD_Mean": fold_df["MAE_USD"].mean(),
            "MAPE_pct_Mean": fold_df["MAPE_pct"].mean(),
            "R2_USD_Mean": fold_df["R2_USD"].mean(),
        }
        results_records.append(summary_row)

        # Fit final estimator on full training data
        full_model = clone(estimator)
        full_model.fit(X, y_log)
        fitted_models[name] = full_model

    summary_df = (
        pd.DataFrame(results_records)
        .sort_values(by="RMSE_log_Mean", ascending=True)
        .reset_index(drop=True)
    )
    return summary_df, oof_predictions, fitted_models


def analyze_residuals(
    y_true_log: pd.Series,
    y_pred_log: np.ndarray,
    df_context: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute detailed residuals and diagnostic statistics to detect heteroscedasticity,
    non-normality, and systematic under/over-prediction across price segments.

    Parameters
    ----------
    y_true_log : pd.Series
        True log1p target values.
    y_pred_log : np.ndarray
        Out-of-fold log1p predictions.
    df_context : pd.DataFrame, optional
        Original/engineered features (e.g., Neighborhood, OverallQual) for error slicing.

    Returns
    -------
    res_df : pd.DataFrame
        Row-level actuals, predictions, residuals (log and USD), and percentage errors.
    diagnostics : dict
        Summary diagnostic statistics (mean residual, skewness, kurtosis, Shapiro p-value,
        and segment bias).
    """
    y_true_log_arr = np.asarray(y_true_log)
    y_pred_log_arr = np.asarray(y_pred_log)

    y_true_usd = np.expm1(y_true_log_arr)
    y_pred_usd = np.expm1(y_pred_log_arr)

    res_log = y_true_log_arr - y_pred_log_arr
    res_usd = y_true_usd - y_pred_usd
    pct_error = (res_usd / y_true_usd) * 100.0
    std_res_log = (res_log - np.mean(res_log)) / np.std(res_log)

    res_df = pd.DataFrame(
        {
            "Actual_Log": y_true_log_arr,
            "Pred_Log": y_pred_log_arr,
            "Residual_Log": res_log,
            "Std_Residual_Log": std_res_log,
            "Actual_USD": y_true_usd,
            "Pred_USD": y_pred_usd,
            "Residual_USD": res_usd,
            "Pct_Error": pct_error,
            "Abs_Pct_Error": np.abs(pct_error),
        }
    )

    # Price tier segmentation to check systematic under/over-prediction
    res_df["Price_Segment"] = pd.qcut(
        res_df["Actual_USD"],
        q=4,
        labels=["Q1: Budget (<$130k)", "Q2: Mid-Low", "Q3: Mid-High", "Q4: Luxury (>$214k)"],
    )

    if df_context is not None:
        for col in ["OverallQual", "GrLivArea", "Neighborhood"]:
            if col in df_context.columns:
                res_df[col] = df_context[col].values

    shapiro_stat, shapiro_p = stats.shapiro(res_log)
    segment_bias = res_df.groupby("Price_Segment", observed=False)["Residual_USD"].mean().to_dict()

    diagnostics = {
        "Mean_Residual_Log": float(np.mean(res_log)),
        "Std_Residual_Log": float(np.std(res_log)),
        "Mean_Residual_USD": float(np.mean(res_usd)),
        "Median_Residual_USD": float(np.median(res_usd)),
        "Residual_Skewness": float(stats.skew(res_log)),
        "Residual_Kurtosis": float(stats.kurtosis(res_log)),
        "Shapiro_Wilk_Stat": float(shapiro_stat),
        "Shapiro_Wilk_p": float(shapiro_p),
        "Outliers_Beyond_3SD": int(np.sum(np.abs(std_res_log) > 3.0)),
        "Segment_Mean_Bias_USD": segment_bias,
    }

    return res_df, diagnostics


def plot_model_comparison(summary_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a 2-panel comparison bar chart of cross-validated RMSE (log scale) and MAE (USD).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Log RMSE with error bars
    sns.barplot(
        data=summary_df,
        x="RMSE_log_Mean",
        y="Model",
        palette="viridis",
        ax=axes[0],
        hue="Model",
        legend=False,
    )
    axes[0].errorbar(
        summary_df["RMSE_log_Mean"],
        range(len(summary_df)),
        xerr=summary_df["RMSE_log_Std"],
        fmt="none",
        c="black",
        capsize=4,
    )
    axes[0].set_title("5-Fold CV RMSE (Log1p Scale — Lower is Better)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("RMSE (log1p SalePrice)")
    axes[0].set_ylabel("")

    # Panel 2: Dollar MAE
    sns.barplot(
        data=summary_df,
        x="MAE_USD_Mean",
        y="Model",
        palette="magma",
        ax=axes[1],
        hue="Model",
        legend=False,
    )
    axes[1].set_title("5-Fold CV Mean Absolute Error (USD — Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("MAE ($ USD)")
    axes[1].set_ylabel("")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_residual_diagnostics(
    res_df: pd.DataFrame,
    model_name: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a 4-panel residual diagnostic suite:
    1. Residuals vs. Predicted Values (Homoscedasticity & Linearity check)
    2. Q-Q Plot of Standardized Residuals (Normality check)
    3. Residual Distribution Histogram + KDE
    4. Systematic Bias by Price Quartile (Under/Over-prediction check)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"Residual Diagnostics — {model_name}", fontsize=15, fontweight="bold", y=1.02)

    # 1. Residuals vs. Fitted (Log scale)
    sns.scatterplot(
        data=res_df,
        x="Pred_Log",
        y="Residual_Log",
        alpha=0.5,
        edgecolor=None,
        color="#2b5c8f",
        ax=axes[0, 0],
    )
    sns.regplot(
        data=res_df,
        x="Pred_Log",
        y="Residual_Log",
        scatter=False,
        lowess=True,
        color="#d95f02",
        line_kws={"linewidth": 2},
        ax=axes[0, 0],
    )
    axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_title("1. Residuals vs. Predicted (Log1p Scale)", fontweight="bold")
    axes[0, 0].set_xlabel("Predicted log1p(SalePrice)")
    axes[0, 0].set_ylabel("Residual (Actual - Predicted)")

    # 2. Normal Q-Q Plot
    stats.probplot(res_df["Std_Residual_Log"], dist="norm", plot=axes[0, 1])
    axes[0, 1].get_lines()[0].set_markerfacecolor("#2b5c8f")
    axes[0, 1].get_lines()[0].set_alpha(0.5)
    axes[0, 1].get_lines()[1].set_color("#d95f02")
    axes[0, 1].set_title("2. Normal Q-Q Plot (Standardized Residuals)", fontweight="bold")

    # 3. Histogram & KDE of Log Residuals
    sns.histplot(res_df["Residual_Log"], kde=True, bins=40, color="#2b5c8f", ax=axes[1, 0])
    axes[1, 0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].axvline(res_df["Residual_Log"].mean(), color="#d95f02", linestyle="-", label="Mean Residual")
    axes[1, 0].set_title("3. Distribution of Log Residuals", fontweight="bold")
    axes[1, 0].set_xlabel("Residual (log1p SalePrice)")
    axes[1, 0].legend()

    # 4. Systematic Under/Over-Prediction by Price Tier (USD)
    sns.boxplot(
        data=res_df,
        x="Price_Segment",
        y="Residual_USD",
        palette="coolwarm",
        hue="Price_Segment",
        legend=False,
        ax=axes[1, 1],
    )
    axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_title("4. Dollar Error by Price Tier (Positive = Under-predicted)", fontweight="bold")
    axes[1, 1].set_xlabel("Actual SalePrice Quartile")
    axes[1, 1].set_ylabel("Residual ($ USD)")
    axes[1, 1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig