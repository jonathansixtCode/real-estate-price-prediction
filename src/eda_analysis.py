"""
Module 2: Exploratory Data Analysis (EDA)
Real Estate House Price Prediction Platform (Ames, Iowa)

Performs comprehensive, reproducible exploratory analysis on the augmented Ames Housing dataset:
1. Target & Feature Distributions + Outlier Detection (IQR, domain leverage points)
2. Correlation & Association Analysis (Pearson & Spearman vs. SalePrice and log1p(SalePrice))
3. Stratified Price Comparisons across Location (Neighborhood), Property Type (BldgType, HouseStyle),
   Property Age cohorts, and Synthetic Renovation Tiers (with Kruskal-Wallis significance tests)
4. Bivariate Relationships (Price vs. Living Area, Overall Quality, Amenity Accessibility, Renovation Cost)
5. Multicollinearity Diagnostics (Pairwise |r| >= 0.70 audit & Intercept-Adjusted VIF)
6. Publication-quality EDA Figure Generation saved to `figures/`
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Module2_EDA")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})
PALETTE_PRIMARY = "#1B365D"
PALETTE_SECONDARY = "#2E86AB"
PALETTE_ACCENT = "#D95F02"
PALETTE_TEAL = "#1B9E77"


def load_augmented_data(
    train_path: str = "data/processed/train_augmented.csv",
    test_path: str = "data/processed/test_augmented.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Module 1 augmented train and test datasets using relative paths."""
    train_file = Path(train_path)
    test_file = Path(test_path)

    if not train_file.exists():
        raise FileNotFoundError(f"Training dataset not found at relative path: {train_file}")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file) if test_file.exists() else pd.DataFrame()

    train_df["LogSalePrice"] = np.log1p(train_df["SalePrice"])
    train_df["PropertyAge"] = (train_df["YrSold"] - train_df["YearBuilt"]).clip(lower=0)
    train_df["YearsSinceRemod"] = (train_df["YrSold"] - train_df["YearRemodAdd"]).clip(lower=0)
    train_df["PricePerSqFt"] = train_df["SalePrice"] / train_df["GrLivArea"]
    train_df["renovation_type_clean"] = train_df["renovation_type"].fillna("None")

    age_bins = [-1, 10, 25, 50, 75, 150]
    age_labels = ["0-10 yrs", "11-25 yrs", "26-50 yrs", "51-75 yrs", "75+ yrs"]
    train_df["AgeCohort"] = pd.cut(train_df["PropertyAge"], bins=age_bins, labels=age_labels)

    logger.info(f"Loaded train_df={train_df.shape} from '{train_file}', test_df={test_df.shape}")
    return train_df, test_df


def analyze_distributions_and_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute distribution diagnostics, skewness/kurtosis, and IQR/domain outlier flags."""
    sp = df["SalePrice"]
    log_sp = df["LogSalePrice"]

    q1, q3 = sp.quantile(0.25), sp.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    iqr_outliers = df[(sp < lower_fence) | (sp > upper_fence)]
    grliv_outliers = df[df["GrLivArea"] > 4000][
        ["Id", "Neighborhood", "OverallQual", "GrLivArea", "SalePrice", "SaleCondition"]
    ]

    key_continuous = [
        "SalePrice", "LogSalePrice", "GrLivArea", "TotalBsmtSF", "LotArea",
        "PropertyAge", "distance_to_school_km", "distance_to_hospital_km",
        "distance_to_transit_km", "amenity_composite_score",
        "renovation_cost_usd", "days_on_market", "mortgage_rate_pct",
        "local_price_index", "unemployment_rate_pct",
    ]
    dist_rows = []
    for col in key_continuous:
        s = df[col].dropna()
        dist_rows.append({
            "Feature": col,
            "Count": int(s.count()),
            "Mean": round(float(s.mean()), 3),
            "Std": round(float(s.std()), 3),
            "Median": round(float(s.median()), 3),
            "IQR": round(float(s.quantile(0.75) - s.quantile(0.25)), 3),
            "Min": round(float(s.min()), 3),
            "Max": round(float(s.max()), 3),
            "Skewness": round(float(s.skew()), 3),
            "Kurtosis": round(float(s.kurt()), 3),
        })

    return {
        "sale_price_summary": {
            "count": int(sp.count()),
            "mean": round(float(sp.mean()), 2),
            "std": round(float(sp.std()), 2),
            "median": round(float(sp.median()), 2),
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "lower_fence": round(float(lower_fence), 2),
            "upper_fence": round(float(upper_fence), 2),
            "min": round(float(sp.min()), 2),
            "max": round(float(sp.max()), 2),
            "raw_skewness": round(float(sp.skew()), 4),
            "raw_kurtosis": round(float(sp.kurt()), 4),
            "log_skewness": round(float(log_sp.skew()), 4),
            "log_kurtosis": round(float(log_sp.kurt()), 4),
            "iqr_outlier_count": int(len(iqr_outliers)),
            "iqr_outlier_pct": round(float(len(iqr_outliers) / len(df) * 100), 2),
        },
        "grliv_extreme_records": grliv_outliers.to_dict(orient="records"),
        "distribution_table": pd.DataFrame(dist_rows),
    }


def compute_correlations(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson and Spearman correlations of numeric predictors with SalePrice & LogSalePrice."""
    exclude = {"Id", "SalePrice", "LogSalePrice", "PricePerSqFt"}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    rows = []
    for col in num_cols:
        valid = df[[col, "SalePrice", "LogSalePrice"]].dropna()
        if len(valid) < 10 or valid[col].nunique() <= 1:
            continue
        r_raw, p_raw = stats.pearsonr(valid[col], valid["SalePrice"])
        rho_raw, _ = stats.spearmanr(valid[col], valid["SalePrice"])
        r_log, _ = stats.pearsonr(valid[col], valid["LogSalePrice"])
        rows.append({
            "Feature": col,
            "Pearson_r_SalePrice": round(float(r_raw), 4),
            "Spearman_rho_SalePrice": round(float(rho_raw), 4),
            "Pearson_r_LogSalePrice": round(float(r_log), 4),
            "Abs_Pearson_r": round(abs(float(r_raw)), 4),
            "p_value": float(p_raw),
        })

    all_corr_df = pd.DataFrame(rows).sort_values("Abs_Pearson_r", ascending=False).reset_index(drop=True)

    synthetic_cols = [
        "distance_to_school_km", "distance_to_hospital_km", "distance_to_transit_km",
        "amenity_composite_score", "has_renovation", "renovation_cost_usd",
        "days_on_market", "mortgage_rate_pct", "local_price_index", "unemployment_rate_pct",
    ]
    synth_corr_df = all_corr_df[all_corr_df["Feature"].isin(synthetic_cols)].reset_index(drop=True)
    return all_corr_df, synth_corr_df


def compare_stratified_price_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Compare SalePrice distributions across Neighborhood, BldgType, HouseStyle, AgeCohort, and Renovation."""
    results: Dict[str, Any] = {}

    group_specs = [
        ("neighborhood", "Neighborhood"),
        ("bldg_type", "BldgType"),
        ("house_style", "HouseStyle"),
        ("age_cohort", "AgeCohort"),
        ("renovation_type", "renovation_type_clean"),
        ("overall_qual", "OverallQual"),
    ]

    for key, col in group_specs:
        agg_df = (
            df.groupby(col, observed=False)["SalePrice"]
            .agg(
                Count="count",
                Mean="mean",
                Median="median",
                Std="std",
                IQR=lambda x: x.quantile(0.75) - x.quantile(0.25),
            )
            .reset_index()
        )
        if key not in {"age_cohort", "overall_qual"}:
            agg_df = agg_df.sort_values("Median", ascending=False).reset_index(drop=True)

        groups = [g["SalePrice"].values for _, g in df.groupby(col, observed=False) if len(g) > 0]
        kw_stat, kw_pval = stats.kruskal(*groups)

        results[key] = {
            "summary_table": agg_df,
            "kruskal_stat": round(float(kw_stat), 4),
            "kruskal_pvalue": float(kw_pval),
        }

    return results


def check_multicollinearity(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify high pairwise correlations (|r| >= 0.70) and compute Intercept-Adjusted
    Variance Inflation Factors (VIF) across core structural & synthetic features.
    """
    exclude = {"Id", "SalePrice", "LogSalePrice", "PricePerSqFt", "PropertyAge", "YearsSinceRemod"}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    corr_matrix = df[num_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: "Abs_Pearson_r"})
        .query("Abs_Pearson_r >= 0.70")
        .sort_values("Abs_Pearson_r", ascending=False)
        .reset_index(drop=True)
    )
    high_pairs["Abs_Pearson_r"] = high_pairs["Abs_Pearson_r"].round(4)

    vif_features = [
        "OverallQual", "GrLivArea", "1stFlrSF", "2ndFlrSF", "TotalBsmtSF",
        "GarageCars", "GarageArea", "FullBath", "TotRmsAbvGrd",
        "YearBuilt", "YearRemodAdd", "GarageYrBlt",
        "distance_to_school_km", "distance_to_hospital_km", "distance_to_transit_km",
        "amenity_composite_score", "renovation_cost_usd", "days_on_market",
        "mortgage_rate_pct", "local_price_index", "unemployment_rate_pct",
    ]
    vif_input = df[vif_features].dropna()
    X_const = sm.add_constant(vif_input)

    vif_rows = []
    for idx, feat in enumerate(vif_features):
        vif_val = variance_inflation_factor(X_const.values, idx + 1)
        if vif_val >= 10.0:
            severity = "Severe (VIF >= 10)"
        elif vif_val >= 5.0:
            severity = "Moderate (5 <= VIF < 10)"
        else:
            severity = "Low (VIF < 5)"
        vif_rows.append({
            "Feature": feat,
            "VIF": round(float(vif_val), 2),
            "Severity": severity,
        })

    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    return high_pairs, vif_df


def generate_eda_figures(df: pd.DataFrame, output_dir: str = "figures") -> List[str]:
    """Generate and save all 6 Module 2 analytical figures using a relative output directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    # --- Figure 1: Target SalePrice vs. Log-Transformed SalePrice Distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df["SalePrice"] / 1000.0, bins=40, kde=True, color=PALETTE_PRIMARY, ax=axes[0])
    axes[0].axvline(df["SalePrice"].mean() / 1000.0, color=PALETTE_ACCENT, linestyle="--", linewidth=1.6,
                    label=f"Mean: ${df['SalePrice'].mean():,.0f}")
    axes[0].axvline(df["SalePrice"].median() / 1000.0, color=PALETTE_TEAL, linestyle="-", linewidth=1.6,
                    label=f"Median: ${df['SalePrice'].median():,.0f}")
    axes[0].set_title(f"Raw SalePrice Distribution (Skew = {df['SalePrice'].skew():.2f})", fontweight="bold")
    axes[0].set_xlabel("Sale Price ($ Thousands)")
    axes[0].set_ylabel("Property Count")
    axes[0].legend()

    sns.histplot(df["LogSalePrice"], bins=40, kde=True, color=PALETTE_SECONDARY, ax=axes[1])
    axes[1].axvline(df["LogSalePrice"].mean(), color=PALETTE_ACCENT, linestyle="--", linewidth=1.6,
                    label=f"Mean: {df['LogSalePrice'].mean():.2f}")
    axes[1].axvline(df["LogSalePrice"].median(), color=PALETTE_TEAL, linestyle="-", linewidth=1.6,
                    label=f"Median: {df['LogSalePrice'].median():.2f}")
    axes[1].set_title(f"Log1p(SalePrice) Distribution (Skew = {df['LogSalePrice'].skew():.2f})", fontweight="bold")
    axes[1].set_xlabel("log(1 + SalePrice)")
    axes[1].set_ylabel("Property Count")
    axes[1].legend()

    plt.suptitle("Figure 1: Target Variable Skewness & Log-Transformation Stabilization", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p1 = str(out_path / "fig1_target_distributions.png")
    plt.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p1)

    # --- Figure 2: Top Feature Correlations with SalePrice (Structural & Synthetic) ---
    all_corr, _ = compute_correlations(df)
    top_corr = pd.concat([
        all_corr.sort_values("Pearson_r_SalePrice", ascending=False).head(12),
        all_corr.sort_values("Pearson_r_SalePrice", ascending=True).head(5),
    ]).drop_duplicates("Feature").sort_values("Pearson_r_SalePrice", ascending=True)

    synth_set = {
        "distance_to_school_km", "distance_to_hospital_km", "distance_to_transit_km",
        "amenity_composite_score", "has_renovation", "renovation_cost_usd",
        "days_on_market", "mortgage_rate_pct", "local_price_index", "unemployment_rate_pct",
    }
    colors = [PALETTE_ACCENT if f in synth_set else PALETTE_PRIMARY for f in top_corr["Feature"]]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(top_corr["Feature"], top_corr["Pearson_r_SalePrice"], color=colors, edgecolor="none", height=0.72)
    ax.axvline(0, color="black", linewidth=0.9)
    for bar, val in zip(bars, top_corr["Pearson_r_SalePrice"]):
        offset = 0.015 if val >= 0 else -0.015
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}", va="center", ha=ha, fontsize=8.5)

    legend_elements = [
        Patch(facecolor=PALETTE_PRIMARY, label="Structural / Core Kaggle Feature"),
        Patch(facecolor=PALETTE_ACCENT, label="Synthetic Contextual Feature (Module 1)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.set_xlim(-0.65, 0.92)
    ax.set_xlabel("Pearson Correlation Coefficient (r) with SalePrice")
    ax.set_title("Figure 2: Strongest Positive & Negative Linear Correlates of SalePrice", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p2 = str(out_path / "fig2_feature_correlations.png")
    plt.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p2)

    # --- Figure 3: Price Patterns Across Location (Neighborhood) ---
    nb_order = df.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(
        data=df, x="Neighborhood", y=df["SalePrice"] / 1000.0,
        order=nb_order, palette="viridis", fliersize=2.5, ax=ax
    )
    ax.axhline(df["SalePrice"].median() / 1000.0, color=PALETTE_ACCENT, linestyle="--", linewidth=1.3,
               label=f"Citywide Median (${df['SalePrice'].median():,.0f})")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Ames Neighborhood (Ordered by Median Sale Price)")
    ax.set_ylabel("Sale Price ($ Thousands)")
    ax.set_title("Figure 3: Sale Price Distribution Across All 25 Ames Neighborhoods", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    p3 = str(out_path / "fig3_price_by_neighborhood.png")
    plt.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p3)

    # --- Figure 4: Price Patterns Across Property Type & Age Cohorts ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    bldg_order = df.groupby("BldgType")["SalePrice"].median().sort_values(ascending=False).index
    sns.boxplot(
        data=df, x="BldgType", y=df["SalePrice"] / 1000.0,
        order=bldg_order, palette=[PALETTE_PRIMARY, PALETTE_SECONDARY, PALETTE_TEAL, "#7570B3", PALETTE_ACCENT],
        ax=axes[0]
    )
    axes[0].set_title("Sale Price by Building Type (BldgType)", fontweight="bold")
    axes[0].set_xlabel("Building Type")
    axes[0].set_ylabel("Sale Price ($ Thousands)")

    sns.boxplot(
        data=df, x="AgeCohort", y=df["SalePrice"] / 1000.0,
        palette="Blues_r", ax=axes[1]
    )
    axes[1].set_title("Sale Price Depreciation Across Property Age Cohorts", fontweight="bold")
    axes[1].set_xlabel("Property Age at Time of Sale")
    axes[1].set_ylabel("Sale Price ($ Thousands)")

    plt.suptitle("Figure 4: Price Stratification by Building Type & Property Age", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p4 = str(out_path / "fig4_price_by_type_and_age.png")
    plt.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p4)

    # --- Figure 5: Bivariate Relationships & Outlier Leverage (Area, Amenities, Renovation) ---
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    normal_mask = df["GrLivArea"] <= 4000
    axes[0].scatter(df.loc[normal_mask, "GrLivArea"], df.loc[normal_mask, "SalePrice"] / 1000.0,
                    alpha=0.45, s=18, color=PALETTE_PRIMARY, label="Standard (<= 4,000 sq ft)")
    axes[0].scatter(df.loc[~normal_mask, "GrLivArea"], df.loc[~normal_mask, "SalePrice"] / 1000.0,
                    alpha=0.95, s=65, color=PALETTE_ACCENT, marker="D", label="Extreme Area (> 4,000 sq ft)")
    sns.regplot(data=df.loc[normal_mask], x="GrLivArea", y=df.loc[normal_mask, "SalePrice"] / 1000.0,
                scatter=False, color=PALETTE_TEAL, ax=axes[0])
    offsets = {524: (-36, 8), 1299: (-42, 8), 692: (-45, 5), 1183: (8, -10)}
    for _, row in df.loc[~normal_mask].iterrows():
        rid = int(row["Id"])
        dx, dy = offsets.get(rid, (-32, 6))
        axes[0].annotate(f"ID {rid}", (row["GrLivArea"], row["SalePrice"] / 1000.0),
                         xytext=(dx, dy), textcoords="offset points", fontsize=8, fontweight="bold", color=PALETTE_ACCENT)
    axes[0].set_title("A. SalePrice vs. Above-Grade Living Area", fontweight="bold")
    axes[0].set_xlabel("GrLivArea (sq ft)")
    axes[0].set_ylabel("Sale Price ($ Thousands)")
    axes[0].legend(fontsize=8)

    sc = axes[1].scatter(df["amenity_composite_score"], df["SalePrice"] / 1000.0,
                         c=df["OverallQual"], cmap="viridis", alpha=0.65, s=20)
    sns.regplot(data=df, x="amenity_composite_score", y=df["SalePrice"] / 1000.0,
                scatter=False, color=PALETTE_ACCENT, ax=axes[1])
    cbar = plt.colorbar(sc, ax=axes[1])
    cbar.set_label("OverallQual (1-10)")
    axes[1].set_title("B. SalePrice vs. Amenity Accessibility Score", fontweight="bold")
    axes[1].set_xlabel("Amenity Composite Score (0-100)")
    axes[1].set_ylabel("Sale Price ($ Thousands)")

    reno_order = ["None", "Cosmetic", "Kitchen_Bath", "Major_Structural"]
    sns.boxplot(data=df, x="renovation_type_clean", y=df["SalePrice"] / 1000.0,
                order=reno_order, palette="Set2", ax=axes[2])
    axes[2].set_title("C. SalePrice by Synthetic Renovation Tier", fontweight="bold")
    axes[2].set_xlabel("Renovation Type")
    axes[2].set_ylabel("Sale Price ($ Thousands)")
    axes[2].tick_params(axis="x", rotation=15)

    plt.suptitle("Figure 5: Key Bivariate Drivers, Spatial Trade-Offs & Structural Outliers", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p5 = str(out_path / "fig5_bivariate_relationships.png")
    plt.savefig(p5, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p5)

    # --- Figure 6: Multicollinearity Heatmap & VIF Diagnostics ---
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6))

    heatmap_cols = [
        "SalePrice", "OverallQual", "GrLivArea", "1stFlrSF", "2ndFlrSF", "TotalBsmtSF",
        "GarageCars", "GarageArea", "TotRmsAbvGrd", "YearBuilt", "GarageYrBlt",
        "distance_to_school_km", "distance_to_hospital_km", "distance_to_transit_km",
        "amenity_composite_score", "mortgage_rate_pct", "local_price_index", "unemployment_rate_pct",
    ]
    corr_sub = df[heatmap_cols].corr()
    mask = np.triu(np.ones_like(corr_sub, dtype=bool))
    sns.heatmap(corr_sub, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 6.5}, cbar_kws={"shrink": 0.8}, ax=axes[0])
    axes[0].set_title("A. Correlation Matrix (Structural & Contextual Clusters)", fontweight="bold")

    _, vif_df = check_multicollinearity(df)
    top_vif = vif_df.head(14).sort_values("VIF", ascending=True)
    vif_colors = [PALETTE_ACCENT if v >= 10 else (PALETTE_SECONDARY if v >= 5 else PALETTE_TEAL) for v in top_vif["VIF"]]
    axes[1].barh(top_vif["Feature"], top_vif["VIF"], color=vif_colors, height=0.7)
    axes[1].axvline(10.0, color=PALETTE_ACCENT, linestyle="--", linewidth=1.3, label="Severe VIF Threshold (10)")
    axes[1].axvline(5.0, color=PALETTE_SECONDARY, linestyle=":", linewidth=1.3, label="Moderate VIF Threshold (5)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Variance Inflation Factor (VIF, Log Scale)")
    axes[1].set_title("B. Intercept-Adjusted VIF Diagnostics (Top 14 Features)", fontweight="bold")
    axes[1].legend(loc="lower right")

    plt.suptitle("Figure 6: Multicollinearity Audit — Structural, Spatial & Macroeconomic Clusters", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p6 = str(out_path / "fig6_multicollinearity_vif.png")
    plt.savefig(p6, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p6)

    logger.info(f"Generated {len(saved_paths)} EDA figures in '{out_path}/'")
    return saved_paths


def run_module2_eda(
    train_path: str = "data/processed/train_augmented.csv",
    test_path: str = "data/processed/test_augmented.csv",
    figures_dir: str = "figures",
) -> Dict[str, Any]:
    """Execute complete Module 2 EDA pipeline using relative paths."""
    train_df, test_df = load_augmented_data(train_path, test_path)
    dist_results = analyze_distributions_and_outliers(train_df)
    all_corr, synth_corr = compute_correlations(train_df)
    strat_results = compare_stratified_price_patterns(train_df)
    high_pairs, vif_df = check_multicollinearity(train_df)
    fig_paths = generate_eda_figures(train_df, output_dir=figures_dir)

    logger.info("Module 2 EDA execution completed successfully.")
    return {
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "distributions": dist_results,
        "all_correlations": all_corr,
        "synthetic_correlations": synth_corr,
        "stratified_patterns": strat_results,
        "high_corr_pairs": high_pairs,
        "vif_table": vif_df,
        "figure_paths": fig_paths,
    }


if __name__ == "__main__":
    results = run_module2_eda()
    print("SalePrice Summary:", json.dumps(results["distributions"]["sale_price_summary"], indent=2))
    print("\nTop 10 Correlations:\n", results["all_correlations"].head(10).to_string(index=False))
    print("\nTop 10 VIF:\n", results["vif_table"].head(10).to_string(index=False))