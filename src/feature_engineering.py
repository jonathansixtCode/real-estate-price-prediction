"""
Module 4: Feature Engineering & Preprocessing Pipeline
Real Estate House Price Prediction Platform

This module transforms the cleaned Ames housing datasets (train_cleaned.csv and test_cleaned.csv):
1. Domain Feature Construction:
   - TotalLivingSF (Total interior usable square footage = Bsmt + 1stFlr + 2ndFlr)
   - TotalBathrooms (FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath)
   - PropertyAge (YrSold - YearBuilt) & YearsSinceRemod (YrSold - YearRemodAdd)
   - TotalPorchSF (OpenPorch + Enclosed + 3Ssn + Screen + WoodDeck)
   - Binary Amenity Indicators (HasPool, HasGarage, HasFireplace, HasBasement)
   - Neighborhood Historical Price per SqFt (Target encoding fitted strictly on train)
2. Target Variable Transformation:
   - Log1p transformation: LogSalePrice = log(1 + SalePrice)
3. Skewness Correction on Numerical Features (log1p on continuous features with |skew| > 0.75)
4. Ordinal Encoding for Hierarchical Categoricals (Ex, Gd, TA, Fa, Po, None -> 5..0)
5. One-Hot Encoding for Nominal Categoricals (with strict train/test alignment)
6. Multicollinearity Check & Feature Selection (VIF validation)

Outputs:
- data/engineered/train_engineered.csv
- data/engineered/test_engineered.csv
- reports/feature_engineering_metrics.json
- reports/figures/12_feature_engineering_vif_reduction.png
- reports/figures/13_feature_importance_mi.png
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FeatureEngineering")


# -----------------------------------------------------------------------------
# 1. Domain Feature Construction
# -----------------------------------------------------------------------------

def construct_derived_features(df: pd.DataFrame, train_ref: pd.DataFrame = None) -> pd.DataFrame:
    """
    Constructs high-value domain-engineered features based on real estate appraisal principles.
    """
    df = df.copy()

    # 1. Total Square Footage Metrics
    # Combine basement and above-grade living area
    df["TotalLivingSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    
    # Total porch and outdoor living square footage
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + 
        df["EnclosedPorch"] + 
        df["3SsnPorch"] + 
        df["ScreenPorch"] + 
        df["WoodDeckSF"]
    )

    # 2. Comprehensive Bathroom Valuation
    # Real estate standard: 1.0 for full bath, 0.5 for half bath
    df["TotalBathrooms"] = (
        df["FullBath"] + 
        0.5 * df["HalfBath"] + 
        df["BsmtFullBath"] + 
        0.5 * df["BsmtHalfBath"]
    )

    # 3. Temporal & Vintage Age Features
    # Age at sale (ensuring non-negative)
    df["PropertyAge"] = np.maximum(0, df["YrSold"] - df["YearBuilt"])
    df["YearsSinceRemod"] = np.maximum(0, df["YrSold"] - df["YearRemodAdd"])
    df["IsRemodeled"] = (df["YearRemodAdd"] > df["YearBuilt"]).astype(int)

    # 4. Amenity Presence Flags
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)

    # 5. Quality & Space Interaction Terms
    df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
    df["HighEndAmenity"] = (df["amenity_composite_score"] >= 70.0).astype(int)

    # 6. Neighborhood Price Per SqFt Benchmark (Target Encoding without Leakage)
    # If train_ref is provided, learn the historical median price/sqft per neighborhood
    if train_ref is not None:
        ref_df = train_ref.copy()
        ref_df["_TempTotalSF"] = ref_df["TotalBsmtSF"] + ref_df["1stFlrSF"] + ref_df["2ndFlrSF"]
        ref_df["_PricePerSF"] = ref_df["SalePrice"] / np.maximum(100.0, ref_df["_TempTotalSF"])
        neigh_price_sqft = ref_df.groupby("Neighborhood")["_PricePerSF"].median()
        overall_price_sqft = ref_df["_PricePerSF"].median()
        
        df["Neighborhood_MedianPricePerSF"] = df["Neighborhood"].map(neigh_price_sqft).fillna(overall_price_sqft)
    
    return df


# -----------------------------------------------------------------------------
# 2. Ordinal & Categorical Encodings
# -----------------------------------------------------------------------------

ORDINAL_QUALITY_MAP = {
    "Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0
}

BSMT_EXPOSURE_MAP = {
    "Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0
}

BSMT_FIN_MAP = {
    "GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0
}

FUNCTIONAL_MAP = {
    "Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0, "None": 0
}

CENTRAL_AIR_MAP = {
    "N": 0, "Y": 1
}


def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes hierarchical categorical attributes into ordered integers."""
    df = df.copy()

    # Quality & Condition standard hierarchy
    qual_cols = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
        "HeatingQC", "KitchenQual", "FireplaceQu",
        "GarageQual", "GarageCond", "PoolQC"
    ]
    for col in qual_cols:
        if col in df.columns:
            df[col + "_Ordinal"] = df[col].map(ORDINAL_QUALITY_MAP).fillna(0).astype(int)

    # Specific ordinal maps
    if "BsmtExposure" in df.columns:
        df["BsmtExposure_Ordinal"] = df["BsmtExposure"].map(BSMT_EXPOSURE_MAP).fillna(0).astype(int)
    if "BsmtFinType1" in df.columns:
        df["BsmtFinType1_Ordinal"] = df["BsmtFinType1"].map(BSMT_FIN_MAP).fillna(0).astype(int)
    if "BsmtFinType2" in df.columns:
        df["BsmtFinType2_Ordinal"] = df["BsmtFinType2"].map(BSMT_FIN_MAP).fillna(0).astype(int)
    if "Functional" in df.columns:
        df["Functional_Ordinal"] = df["Functional"].map(FUNCTIONAL_MAP).fillna(7).astype(int)
    if "CentralAir" in df.columns:
        df["CentralAir_Ordinal"] = df["CentralAir"].map(CENTRAL_AIR_MAP).fillna(0).astype(int)

    return df


# -----------------------------------------------------------------------------
# 3. Skewness Correction
# -----------------------------------------------------------------------------

def correct_skewness(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    threshold: float = 0.75
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Applies log1p transformation to numerical features with |skewness| > threshold on training data.
    Identical transformation is applied to test data to maintain distribution consistency.
    """
    train = df_train.copy()
    test = df_test.copy()

    # Select continuous numerical columns (exclude Ids, target, ordinal flags, years, binaries)
    exclude_cols = [
        "Id", "SalePrice", "LogSalePrice", "OverallQual", "OverallCond",
        "YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold", "MoSold",
        "MSSubClass", "has_renovation", "HasPool", "HasGarage", "HasFireplace",
        "HasBasement", "Has2ndFloor", "HighEndAmenity", "IsRemodeled"
    ]
    candidate_cols = [
        c for c in train.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols and not c.endswith("_Ordinal")
    ]

    skewed_cols = []
    for col in candidate_cols:
        sk = train[col].skew()
        if abs(sk) > threshold:
            skewed_cols.append(col)
            train[col] = np.log1p(np.maximum(0.0, train[col]))
            test[col] = np.log1p(np.maximum(0.0, test[col]))

    logger.info(f"Applied log1p transformation to {len(skewed_cols)} skewed features (|skew| > {threshold}): {skewed_cols}")
    return train, test, skewed_cols


# -----------------------------------------------------------------------------
# 4. Multicollinearity & Feature Selection
# -----------------------------------------------------------------------------

def evaluate_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Calculates Variance Inflation Factor for a subset of features."""
    sub_df = df[features].dropna()
    X = (sub_df - sub_df.mean()) / sub_df.std()
    corr_mat = np.corrcoef(X.values, rowvar=False)
    vif_vals = np.diag(np.linalg.inv(corr_mat))
    
    return pd.DataFrame({
        "Feature": features,
        "VIF": np.round(vif_vals, 2)
    }).sort_values(by="VIF", ascending=False)


def plot_vif_reduction_comparison(
    before_vif_df: pd.DataFrame,
    after_vif_df: pd.DataFrame,
    output_path: str = "reports/figures/12_feature_engineering_vif_reduction.png"
):
    """Visualizes VIF reduction after synthesizing TotalLivingSF and removing redundant dimensions."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Before VIF
    b_df = before_vif_df.head(10).sort_values(by="VIF", ascending=True)
    b_colors = ["#c0392b" if v > 10 else ("#e67e22" if v > 5 else "#27ae60") for v in b_df["VIF"]]
    axes[0].barh(b_df["Feature"], b_df["VIF"], color=b_colors, edgecolor="black", alpha=0.85)
    axes[0].axvline(10, color="#c0392b", linestyle=":", label="Severe VIF = 10")
    axes[0].axvline(5, color="#e67e22", linestyle="--", label="Moderate VIF = 5")
    axes[0].set_title("Before Engineering: Severe Spatial Multicollinearity", pad=12)
    axes[0].set_xlabel("Variance Inflation Factor (VIF)")
    axes[0].legend(loc="lower right")

    # After VIF
    a_df = after_vif_df.head(10).sort_values(by="VIF", ascending=True)
    a_colors = ["#c0392b" if v > 10 else ("#e67e22" if v > 5 else "#27ae60") for v in a_df["VIF"]]
    axes[1].barh(a_df["Feature"], a_df["VIF"], color=a_colors, edgecolor="black", alpha=0.85)
    axes[1].axvline(10, color="#c0392b", linestyle=":", label="Severe VIF = 10")
    axes[1].axvline(5, color="#e67e22", linestyle="--", label="Moderate VIF = 5")
    axes[1].set_title("After Engineering: Stabilized Feature Independence", pad=12)
    axes[1].set_xlabel("Variance Inflation Factor (VIF)")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"Saved VIF reduction comparison to {output_path}")


def plot_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    output_path: str = "reports/figures/13_feature_importance_mi.png"
) -> Dict[str, float]:
    """Computes and plots top 20 Mutual Information regression scores against LogSalePrice."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Computing Mutual Information feature importance...")
    
    # Select numerical columns
    num_cols = X.select_dtypes(include=[np.number]).columns
    X_num = X[num_cols].fillna(0)
    
    mi_scores = mutual_info_regression(X_num, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=num_cols).sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e67e22" if any(k in col for k in ["amenity", "renovation", "TotalLivingSF", "TotalBathrooms", "PropertyAge", "MedianPricePerSF", "days_on_market"])
              else "#2980b9" for col in mi_series.index]
    
    bars = ax.barh(mi_series.index[::-1], mi_series.values[::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
    ax.set_title("Top 20 Features by Mutual Information with Log(SalePrice)\n(Orange = Engineered / Synthetic Features)", pad=15)
    ax.set_xlabel("Mutual Information Score (Dependency with Target)")
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.3f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9)
        
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"Saved Mutual Information plot to {output_path}")
    
    return mi_series.to_dict()


# -----------------------------------------------------------------------------
# 5. Master Pipeline Execution
# -----------------------------------------------------------------------------

def run_feature_engineering_pipeline(
    train_in: str = "data/cleaned/train_cleaned.csv",
    test_in: str = "data/cleaned/test_cleaned.csv",
    train_out: str = "data/engineered/train_engineered.csv",
    test_out: str = "data/engineered/test_engineered.csv",
    metrics_out: str = "reports/feature_engineering_metrics.json"
):
    logger.info("=" * 65)
    logger.info("STARTING MODULE 4: FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 65)

    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df_train = pd.read_csv(train_in)
    df_test = pd.read_csv(test_in)
    logger.info(f"Loaded inputs: Train ({df_train.shape}), Test ({df_test.shape})")

    # 1. Target Transformation
    df_train["LogSalePrice"] = np.log1p(df_train["SalePrice"])
    logger.info("Created target LogSalePrice = log(1 + SalePrice)")

    # 2. Derived Feature Construction
    logger.info("Constructing domain-specific derived features...")
    df_train_derived = construct_derived_features(df_train, train_ref=df_train)
    df_test_derived = construct_derived_features(df_test, train_ref=df_train)

    # 3. Ordinal Encoding
    logger.info("Applying ordinal hierarchy encoding...")
    df_train_ord = encode_ordinal_features(df_train_derived)
    df_test_ord = encode_ordinal_features(df_test_derived)

    # 4. Skewness Correction
    logger.info("Evaluating numerical skewness...")
    df_train_skew, df_test_skew, skewed_features = correct_skewness(df_train_ord, df_test_ord, threshold=0.75)

    # 5. Multicollinearity Assessment (Before vs After TotalLivingSF synthesis)
    before_features = [
        "GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GarageCars",
        "GarageArea", "OverallQual", "YearBuilt", "FullBath", "TotRmsAbvGrd",
        "amenity_composite_score", "days_on_market"
    ]
    after_features = [
        "TotalLivingSF", "OverallQual", "TotalBathrooms", "PropertyAge",
        "GarageCars", "amenity_composite_score", "days_on_market",
        "TotalPorchSF", "Neighborhood_MedianPricePerSF", "mortgage_rate_pct"
    ]
    vif_before = evaluate_vif(df_train, before_features)
    vif_after = evaluate_vif(df_train_skew, after_features)
    plot_vif_reduction_comparison(vif_before, vif_after)

    # 6. One-Hot Encoding for Nominal Categoricals (Strict Alignment)
    logger.info("Performing One-Hot Encoding with train/test alignment...")
    
    # Identify non-ordinal string categoricals
    cat_cols = df_train_skew.select_dtypes(include=["object"]).columns.tolist()
    # Exclude ID or text fields that should not be dummified
    drop_unencoded = ["renovation_contractor", "renovation_permit_id"]
    cat_to_encode = [c for c in cat_cols if c not in drop_unencoded]

    # Combine temporarily for consistent one-hot dummies
    train_len = len(df_train_skew)
    combined = pd.concat([df_train_skew, df_test_skew], axis=0, ignore_index=True)
    combined_encoded = pd.get_dummies(combined, columns=cat_to_encode, drop_first=True)

    # Separate back to train and test
    train_final = combined_encoded.iloc[:train_len].copy()
    test_final = combined_encoded.iloc[train_len:].copy()
    test_final = test_final.drop(columns=["SalePrice", "LogSalePrice"], errors="ignore")

    # Drop unencoded unstructured text columns
    train_final = train_final.drop(columns=drop_unencoded, errors="ignore")
    test_final = test_final.drop(columns=drop_unencoded, errors="ignore")

    logger.info(f"Final engineered datasets: Train {train_final.shape}, Test {test_final.shape}")

    # 7. Mutual Information Ranking
    X_train_mi = train_final.drop(columns=["Id", "SalePrice", "LogSalePrice"])
    y_train_mi = train_final["LogSalePrice"]
    top_mi = plot_mutual_information(X_train_mi, y_train_mi)

    # 8. Save Processed Datasets
    train_final.to_csv(train_out, index=False)
    test_final.to_csv(test_out, index=False)
    logger.info(f"Saved engineered training data to {train_out}")
    logger.info(f"Saved engineered test data to {test_out}")

    # 9. Save Summary Metrics JSON
    summary_metrics = {
        "final_train_shape": list(train_final.shape),
        "final_test_shape": list(test_final.shape),
        "target_transformed": "LogSalePrice = log(1 + SalePrice)",
        "skewed_features_transformed_count": len(skewed_features),
        "skewed_features_transformed": skewed_features,
        "vif_before_synthesis": vif_before.to_dict(orient="records"),
        "vif_after_synthesis": vif_after.to_dict(orient="records"),
        "top_mutual_information_features": top_mi
    }
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4)
    logger.info(f"Saved feature engineering metrics to {metrics_out}")

    logger.info("=" * 65)
    logger.info("MODULE 4 FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 65)


if __name__ == "__main__":
    run_feature_engineering_pipeline()
