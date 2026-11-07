"""
Module 3: Data Cleaning Pipeline
Real Estate House Price Prediction Platform

This module implements the end-to-end data cleaning pipeline for the augmented
Ames housing dataset (train_augmented.csv and test_augmented.csv):
1. Duplicate Identification and Handling
2. Domain-Specific Missing Value Imputation (Structural vs. Non-Structural)
3. Categorical Standardization and Typo Rectification
4. Outlier Diagnostics and Treatment (Dean De Cock anomalies & entry typos)
5. Generation of Before/After Quality Comparison Metrics

Output Datasets:
- data/cleaned/train_cleaned.csv
- data/cleaned/test_cleaned.csv
- reports/cleaning_before_after_metrics.json
- reports/figures/11_cleaning_before_after.png
"""

import os
import sys
import json
import logging
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataCleaning")


# -----------------------------------------------------------------------------
# 1. Cleaning Configuration & Domain Dictionaries
# -----------------------------------------------------------------------------

# Categorical features where 'NA' means the physical feature does NOT exist
CATEGORICAL_NONE_VARS = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"
]

# Numerical features where 'NA' means 0 (no basement, no garage, etc.)
NUMERICAL_ZERO_VARS = [
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
    "MasVnrArea"
]

# Categorical features where 'NA' is a genuine missing entry, imputed by mode
CATEGORICAL_MODE_VARS = [
    "Electrical", "MSZoning", "Utilities", "Exterior1st",
    "Exterior2nd", "KitchenQual", "Functional", "SaleType"
]


# -----------------------------------------------------------------------------
# 2. Cleaning Steps
# -----------------------------------------------------------------------------

def audit_duplicates(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, int]:
    """Identifies exact and feature-level duplicate rows."""
    logger.info("Auditing duplicate records...")
    # Exclude 'Id' and target 'SalePrice' when checking feature duplicates
    feature_cols = [c for c in df_train.columns if c not in ["Id", "SalePrice"]]
    
    exact_dups_train = int(df_train.duplicated(subset=feature_cols).sum())
    exact_dups_test = int(df_test.duplicated(subset=feature_cols).sum())
    
    logger.info(f"Duplicate check: {exact_dups_train} duplicates in train, {exact_dups_test} in test.")
    return {"train_duplicates": exact_dups_train, "test_duplicates": exact_dups_test}


def fix_data_entry_typos(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Fixes known data-entry typos and inconsistent values."""
    actions = []
    df = df.copy()

    # Typo 1: GarageYrBlt = 2207 (Observation Id 2593 in test set)
    # The garage was built in 2007 (matching YearBuilt = 2006/2007), but recorded as 2207
    if "GarageYrBlt" in df.columns:
        invalid_garage_mask = df["GarageYrBlt"] > 2026
        if invalid_garage_mask.any():
            count = int(invalid_garage_mask.sum())
            df.loc[invalid_garage_mask, "GarageYrBlt"] = df.loc[invalid_garage_mask, "YearBuilt"]
            actions.append(f"Fixed {count} impossible future GarageYrBlt (> 2026) by setting to YearBuilt.")
            logger.info(actions[-1])

    # Typo 2: MSZoning category consistency ('C (all)' -> 'C')
    if "MSZoning" in df.columns:
        zoning_mask = df["MSZoning"] == "C (all)"
        if zoning_mask.any():
            count = int(zoning_mask.sum())
            df.loc[zoning_mask, "MSZoning"] = "C"
            actions.append(f"Standardized {count} MSZoning 'C (all)' entries to 'C'.")
            logger.info(actions[-1])

    # Typo 3: Strip extraneous whitespace from categorical strings
    cat_cols = df.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip()

    return df, actions


def handle_missing_values(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Imputes missing values using domain rules, neighborhood-grouped medians, and modes.
    Learns imputation statistics exclusively on training data to prevent data leakage.
    """
    logger.info("Handling missing values across train and test sets...")
    imputation_log = {}
    train = df_train.copy()
    test = df_test.copy()

    # 1. Structural Categoricals: 'NA' -> 'None'
    for col in CATEGORICAL_NONE_VARS:
        if col in train.columns:
            train[col] = train[col].fillna("None")
        if col in test.columns:
            test[col] = test[col].fillna("None")
    imputation_log["structural_categoricals_to_none"] = CATEGORICAL_NONE_VARS

    # 2. Structural Numericals: 'NA' -> 0.0
    for col in NUMERICAL_ZERO_VARS:
        if col in train.columns:
            train[col] = train[col].fillna(0.0)
        if col in test.columns:
            test[col] = test[col].fillna(0.0)
    imputation_log["structural_numericals_to_zero"] = NUMERICAL_ZERO_VARS

    # 3. GarageYrBlt: Set missing garage year to 0 (indicating no garage)
    if "GarageYrBlt" in train.columns:
        train["GarageYrBlt"] = train["GarageYrBlt"].fillna(0.0)
    if "GarageYrBlt" in test.columns:
        test["GarageYrBlt"] = test["GarageYrBlt"].fillna(0.0)

    # 4. MasVnrType: If MasVnrArea is 0, type is 'None'
    if "MasVnrType" in train.columns:
        train["MasVnrType"] = train["MasVnrType"].fillna("None")
        train.loc[train["MasVnrArea"] == 0, "MasVnrType"] = "None"
    if "MasVnrType" in test.columns:
        test["MasVnrType"] = test["MasVnrType"].fillna("None")
        test.loc[test["MasVnrArea"] == 0, "MasVnrType"] = "None"

    # 5. LotFrontage: Impute by Neighborhood-specific median (prevents data leakage: fit on train)
    neighborhood_medians = train.groupby("Neighborhood")["LotFrontage"].median()
    overall_median = train["LotFrontage"].median()
    imputation_log["lot_frontage_neighborhood_medians"] = neighborhood_medians.to_dict()

    train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(
        lambda s: s.fillna(s.median())
    ).fillna(overall_median)

    # Map train neighborhood medians onto test set
    test["LotFrontage"] = test["LotFrontage"].fillna(
        test["Neighborhood"].map(neighborhood_medians)
    ).fillna(overall_median)
    imputation_log["lot_frontage_imputation"] = "Neighborhood-grouped median from training set"

    # 6. Genuine Mode Categoricals (fit on train, apply to train & test)
    mode_imputations = {}
    for col in CATEGORICAL_MODE_VARS:
        if col in train.columns:
            mode_val = train[col].mode()[0]
            mode_imputations[col] = mode_val
            train[col] = train[col].fillna(mode_val)
            if col in test.columns:
                test[col] = test[col].fillna(mode_val)
    imputation_log["mode_imputations"] = mode_imputations

    # Verify no missing values remain
    train_nulls = int(train.drop(columns=["SalePrice"], errors="ignore").isna().sum().sum())
    test_nulls = int(test.isna().sum().sum())
    logger.info(f"Post-imputation check: {train_nulls} nulls in train features, {test_nulls} nulls in test.")

    return train, test, imputation_log


def treat_outliers(df_train: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Removes severe bivariate outliers identified in Module 2 (Dean De Cock anomalies).
    Specifically: Properties with GrLivArea > 4000 sq ft and SalePrice < $300,000.
    """
    logger.info("Treating severe leverage outliers in training set...")
    initial_len = len(df_train)

    # Identify outliers: Id 524 and 1299
    outlier_mask = (df_train["GrLivArea"] > 4000) & (df_train["SalePrice"] < 300000)
    outlier_ids = df_train.loc[outlier_mask, "Id"].tolist()

    df_cleaned = df_train[~outlier_mask].copy()
    dropped_count = initial_len - len(df_cleaned)

    logger.info(f"Dropped {dropped_count} extreme leverage outliers: Record IDs {outlier_ids}")
    
    outlier_report = {
        "dropped_records_count": dropped_count,
        "dropped_record_ids": outlier_ids,
        "initial_rows": initial_len,
        "retained_rows": len(df_cleaned)
    }
    return df_cleaned, outlier_report


def generate_before_after_visualization(
    raw_train: pd.DataFrame,
    clean_train: pd.DataFrame,
    output_path: str = "reports/figures/11_cleaning_before_after.png"
):
    """Creates publication-quality visual comparisons of data state before and after cleaning."""
    logger.info(f"Generating Before/After comparison visual: {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Missing Values Count (Top 10 missing features before vs after)
    raw_missing = raw_train.isna().sum().sort_values(ascending=False).head(10)
    clean_missing = clean_train[raw_missing.index].isna().sum()

    x = np.arange(len(raw_missing))
    width = 0.38
    axes[0, 0].bar(x - width/2, raw_missing.values, width, label="Before Cleaning (Raw)", color="#e74c3c", alpha=0.85)
    axes[0, 0].bar(x + width/2, clean_missing.values, width, label="After Cleaning (Cleaned)", color="#27ae60", alpha=0.85)
    axes[0, 0].set_title("Missing Value Imputation: Top 10 Columns", pad=12)
    axes[0, 0].set_ylabel("Missing Count")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(raw_missing.index, rotation=45, ha="right")
    axes[0, 0].legend()

    # 2. Outlier Treatment: GrLivArea vs SalePrice Before Cleaning
    sns.scatterplot(data=raw_train, x="GrLivArea", y="SalePrice", ax=axes[0, 1], alpha=0.5, color="#7f8c8d")
    outliers = raw_train[(raw_train["GrLivArea"] > 4000) & (raw_train["SalePrice"] < 300000)]
    axes[0, 1].scatter(outliers["GrLivArea"], outliers["SalePrice"], color="#e74c3c", s=120, edgecolor="black", label="Dropped Outliers (Id 524, 1299)")
    axes[0, 1].set_title("Before Cleaning: Severe Outliers Distorting Slope", pad=12)
    axes[0, 1].set_xlabel("Living Area (sq ft)")
    axes[0, 1].set_ylabel("Sale Price ($)")
    axes[0, 1].yaxis.set_major_formatter('${x:,.0f}')
    axes[0, 1].legend()

    # 3. Outlier Treatment: GrLivArea vs SalePrice After Cleaning
    sns.scatterplot(data=clean_train, x="GrLivArea", y="SalePrice", ax=axes[1, 0], alpha=0.5, color="#2980b9")
    sns.regplot(data=clean_train, x="GrLivArea", y="SalePrice", ax=axes[1, 0], scatter=False, color="#27ae60", line_kws={"linewidth": 2, "label": "Clean Linear Fit"})
    axes[1, 0].set_title("After Cleaning: Clean Monotonic Linear Trajectory", pad=12)
    axes[1, 0].set_xlabel("Living Area (sq ft)")
    axes[1, 0].set_ylabel("Sale Price ($)")
    axes[1, 0].yaxis.set_major_formatter('${x:,.0f}')
    axes[1, 0].legend()

    # 4. LotFrontage Distribution Before (with NaNs) vs After Imputation
    raw_frontage = raw_train["LotFrontage"].dropna()
    clean_frontage = clean_train["LotFrontage"]
    sns.kdeplot(raw_frontage, ax=axes[1, 1], color="#e74c3c", linestyle="--", linewidth=2, label=f"Before Imputation (n={len(raw_frontage)})")
    sns.kdeplot(clean_frontage, ax=axes[1, 1], color="#27ae60", linewidth=2, label=f"After Imputation (n={len(clean_frontage)})")
    axes[1, 1].set_title("LotFrontage Distribution Preservation (Neighborhood Median)", pad=12)
    axes[1, 1].set_xlabel("Lot Frontage (Linear Feet)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved Before/After visualization.")


# -----------------------------------------------------------------------------
# 3. Main Pipeline Controller
# -----------------------------------------------------------------------------

def run_data_cleaning_pipeline(
    train_in: str = "data/processed/train_augmented.csv",
    test_in: str = "data/processed/test_augmented.csv",
    train_out: str = "data/cleaned/train_cleaned.csv",
    test_out: str = "data/cleaned/test_cleaned.csv",
    metrics_out: str = "reports/cleaning_before_after_metrics.json"
):
    logger.info("=" * 65)
    logger.info("STARTING MODULE 3: DATA CLEANING PIPELINE")
    logger.info("=" * 65)

    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df_train_raw = pd.read_csv(train_in)
    df_test_raw = pd.read_csv(test_in)
    logger.info(f"Loaded raw inputs: Train ({df_train_raw.shape}), Test ({df_test_raw.shape})")

    # Metrics tracker
    metrics = {
        "raw_train_shape": list(df_train_raw.shape),
        "raw_test_shape": list(df_test_raw.shape),
        "raw_train_total_missing": int(df_train_raw.isna().sum().sum()),
        "raw_test_total_missing": int(df_test_raw.isna().sum().sum()),
    }

    # Step 1: Duplicate Audit
    dup_stats = audit_duplicates(df_train_raw, df_test_raw)
    metrics["duplicate_audit"] = dup_stats

    # Step 2: Fix Data Entry Typos & Inconsistencies
    df_train_typo, train_actions = fix_data_entry_typos(df_train_raw)
    df_test_typo, test_actions = fix_data_entry_typos(df_test_raw)
    metrics["typo_actions"] = {"train": train_actions, "test": test_actions}

    # Step 3: Handle Missing Values (fit on train, transform train & test)
    df_train_imputed, df_test_imputed, impute_log = handle_missing_values(df_train_typo, df_test_typo)
    metrics["imputation_summary"] = impute_log

    # Step 4: Outlier Treatment (train set only)
    df_train_cleaned, outlier_log = treat_outliers(df_train_imputed)
    metrics["outlier_treatment"] = outlier_log

    # Final Verification
    train_features_null = int(df_train_cleaned.drop(columns=["SalePrice"], errors="ignore").isna().sum().sum())
    test_features_null = int(df_test_imputed.isna().sum().sum())
    
    metrics["clean_train_shape"] = list(df_train_cleaned.shape)
    metrics["clean_test_shape"] = list(df_test_imputed.shape)
    metrics["clean_train_missing_features"] = train_features_null
    metrics["clean_test_missing_features"] = test_features_null

    assert train_features_null == 0, f"Error: {train_features_null} missing values remain in train!"
    assert test_features_null == 0, f"Error: {test_features_null} missing values remain in test!"

    # Save cleaned datasets
    df_train_cleaned.to_csv(train_out, index=False)
    df_test_imputed.to_csv(test_out, index=False)
    logger.info(f"Saved cleaned train dataset ({df_train_cleaned.shape}) to {train_out}")
    logger.info(f"Saved cleaned test dataset ({df_test_imputed.shape}) to {test_out}")

    # Generate visual figure
    generate_before_after_visualization(df_train_raw, df_train_cleaned)

    # Save metrics JSON
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved cleaning metrics to {metrics_out}")

    logger.info("=" * 65)
    logger.info("MODULE 3 DATA CLEANING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 65)


if __name__ == "__main__":
    run_data_cleaning_pipeline()
