"""
Module 2: Exploratory Data Analysis (EDA) Pipeline
Real Estate House Price Prediction Platform

This script performs comprehensive exploratory data analysis on the augmented
Ames housing training dataset (data/processed/train_augmented.csv):
1. Target Distribution & Transformation Analysis (SalePrice & Log-SalePrice)
2. Univariate & Bivariate Outlier Diagnostics (GrLivArea, LotArea, etc.)
3. Correlation & Feature Importance Analysis (Pearson, Spearman)
4. Spatial & Locational Price Patterns (Neighborhoods, Amenity Scores)
5. Architectural & Temporal Patterns (BldgType, HouseStyle, Age, Renovation)
6. Macroeconomic Influences (Mortgage Rate, HPI over 2006-2010)
7. Multicollinearity Assessment (Correlation Matrix & Variance Inflation Factors - VIF)

All generated publication-quality figures are exported to reports/figures/.
Summary statistical tables are written to reports/eda_summary_metrics.json.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

logger = logging.getLogger("EDA")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_DIR = "reports/figures"
DATA_PATH = "data/processed/train_augmented.csv"


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)


def plot_target_distribution(df: pd.DataFrame) -> dict:
    """Analyzes skewness, kurtosis, and normality of SalePrice before and after log-transform."""
    logger.info("Generating Figure 1: Target Variable Distribution...")
    raw_price = df["SalePrice"]
    log_price = np.log1p(raw_price)

    raw_skew = float(raw_price.skew())
    raw_kurt = float(raw_price.kurt())
    log_skew = float(log_price.skew())
    log_kurt = float(log_price.kurt())

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Raw SalePrice Histogram + KDE
    sns.histplot(raw_price, kde=True, ax=axes[0, 0], color="#2b5c8f", bins=50)
    axes[0, 0].set_title(f"Raw SalePrice Distribution\nSkewness: {raw_skew:.2f} | Kurtosis: {raw_kurt:.2f}")
    axes[0, 0].set_xlabel("Sale Price ($)")
    axes[0, 0].xaxis.set_major_formatter('${x:,.0f}')

    # 2. Raw SalePrice Q-Q Plot
    stats.probplot(raw_price, dist="norm", plot=axes[0, 1])
    axes[0, 1].get_lines()[0].set_color("#2b5c8f")
    axes[0, 1].get_lines()[1].set_color("#d9534f")
    axes[0, 1].set_title("Normal Q-Q Plot: Raw SalePrice")

    # 3. Log1p SalePrice Histogram + KDE
    sns.histplot(log_price, kde=True, ax=axes[1, 0], color="#28a745", bins=50)
    axes[1, 0].set_title(f"Log1p-Transformed SalePrice\nSkewness: {log_skew:.2f} | Kurtosis: {log_kurt:.2f}")
    axes[1, 0].set_xlabel("Log(1 + Sale Price)")

    # 4. Log1p SalePrice Q-Q Plot
    stats.probplot(log_price, dist="norm", plot=axes[1, 1])
    axes[1, 1].get_lines()[0].set_color("#28a745")
    axes[1, 1].get_lines()[1].set_color("#d9534f")
    axes[1, 1].set_title("Normal Q-Q Plot: Log1p-Transformed SalePrice")

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "01_target_distribution.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "raw_skewness": raw_skew,
        "raw_kurtosis": raw_kurt,
        "log_skewness": log_skew,
        "log_kurtosis": log_kurt,
        "median_saleprice": float(raw_price.median()),
        "mean_saleprice": float(raw_price.mean()),
        "std_saleprice": float(raw_price.std())
    }


def plot_correlations(df: pd.DataFrame) -> dict:
    """Computes Pearson correlations with SalePrice and Log-SalePrice."""
    logger.info("Generating Figures 2 & 3: Correlation Analysis...")
    num_df = df.select_dtypes(include=[np.number]).drop(columns=["Id"], errors="ignore")
    
    # Compute correlation with raw and log price
    df_temp = num_df.copy()
    df_temp["LogSalePrice"] = np.log1p(df_temp["SalePrice"])
    corr_series = df_temp.corr()["SalePrice"].sort_values(ascending=False)
    corr_log = df_temp.corr()["LogSalePrice"].sort_values(ascending=False)

    # Top 15 positive and top 5 negative correlations (excluding target itself)
    top_pos = corr_series.drop(labels=["SalePrice", "LogSalePrice"]).head(15)
    
    # Figure 2: Horizontal Bar Plot of Top Correlated Features
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2b5c8f" if "amenity" not in c and "renovation" not in c and "market" not in c and "price_index" not in c and "rate" not in c 
              else "#e67e22" for c in top_pos.index]
    bars = ax.barh(top_pos.index[::-1], top_pos.values[::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
    ax.set_title("Top Numerical Features Correlated with SalePrice\n(Orange = Synthetic Contextual Features)", pad=15)
    ax.set_xlabel("Pearson Correlation Coefficient ($r$)")
    ax.set_xlim(0, 0.85)
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.3f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_top_correlations.png"), bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Heatmap of Key Multidimensional Drivers
    key_cols = [
        "SalePrice", "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF",
        "1stFlrSF", "FullBath", "YearBuilt", "amenity_composite_score",
        "renovation_cost_usd", "days_on_market", "local_price_index", "mortgage_rate_pct"
    ]
    key_corr = df[key_cols].corr()
    
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(key_corr, dtype=bool))
    sns.heatmap(key_corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.6, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap: Core Physical & Contextual Features", pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_correlation_heatmap.png"), bbox_inches="tight")
    plt.close(fig)

    return {
        "top_correlated_features": top_pos.to_dict()
    }


def plot_neighborhood_patterns(df: pd.DataFrame):
    """Visualizes median price distribution and dispersion across Ames neighborhoods."""
    logger.info("Generating Figure 4: Neighborhood Spatial Patterns...")
    neigh_order = df.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=df, x="Neighborhood", y="SalePrice", order=neigh_order, ax=ax, palette="Blues_r", fliersize=3)
    ax.set_title("Property Sale Price Distribution across Ames Neighborhoods\n(Sorted by Median Sale Price)", pad=15)
    ax.set_xlabel("Neighborhood Code")
    ax.set_ylabel("Sale Price ($)")
    ax.yaxis.set_major_formatter('${x:,.0f}')
    ax.tick_params(axis="x", rotation=45)

    # Add reference median line
    overall_median = df["SalePrice"].median()
    ax.axhline(overall_median, color="#d9534f", linestyle="--", linewidth=1.5,
               label=f"Overall Ames Median: ${overall_median:,.0f}")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_price_by_neighborhood.png"), bbox_inches="tight")
    plt.close(fig)


def plot_structural_and_vintage_patterns(df: pd.DataFrame):
    """Examines price dynamics across Building Types, House Styles, and Property Age."""
    logger.info("Generating Figures 5 & 6: Property Typology & Vintage Trends...")
    
    # Figure 5: BldgType & HouseStyle
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    bldg_order = df.groupby("BldgType")["SalePrice"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="BldgType", y="SalePrice", order=bldg_order, ax=axes[0], palette="Set2")
    axes[0].set_title("Sale Price by Building Type (BldgType)")
    axes[0].set_xlabel("Building Type")
    axes[0].set_ylabel("Sale Price ($)")
    axes[0].yaxis.set_major_formatter('${x:,.0f}')
    
    style_order = df.groupby("HouseStyle")["SalePrice"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="HouseStyle", y="SalePrice", order=style_order, ax=axes[1], palette="Set2")
    axes[1].set_title("Sale Price by Dwelling Style (HouseStyle)")
    axes[1].set_xlabel("House Style")
    axes[1].set_ylabel("")
    axes[1].yaxis.set_major_formatter('${x:,.0f}')
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_price_by_building_type_and_style.png"), bbox_inches="tight")
    plt.close(fig)

    # Figure 6: Age, Renovation & Depreciation Curve
    df_plot = df.copy()
    df_plot["AgeAtSale"] = df_plot["YrSold"] - df_plot["YearBuilt"]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    palette = {"None": "#7f8c8d", "Cosmetic": "#3498db", "Kitchen_Bath": "#e67e22", "Major_Structural": "#e74c3c"}
    sns.scatterplot(data=df_plot, x="AgeAtSale", y="SalePrice", hue="renovation_type",
                    palette=palette, alpha=0.65, s=45, ax=ax)
    
    # Polynomial trend line (degree 2)
    sns.regplot(data=df_plot, x="AgeAtSale", y="SalePrice", scatter=False, ax=ax,
                color="#2c3e50", order=2, line_kws={"linewidth": 2.5, "label": "Nonlinear Age Depreciation Trend"})
    
    ax.set_title("Property Age at Sale vs. Sale Price Colored by Renovation Type", pad=15)
    ax.set_xlabel("Property Age at Time of Sale (Years)")
    ax.set_ylabel("Sale Price ($)")
    ax.yaxis.set_major_formatter('${x:,.0f}')
    ax.legend(title="Renovation Scope", loc="upper right")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "06_price_vs_age_and_renovation.png"), bbox_inches="tight")
    plt.close(fig)


def plot_key_drivers_and_outliers(df: pd.DataFrame) -> dict:
    """Visualizes key bivariate relationships and pinpoints severe outliers."""
    logger.info("Generating Figures 7 & 8: Bivariate Drivers & Outlier Diagnostics...")
    
    # Figure 7: 2x2 Grid of Key Predictors
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # 1. GrLivArea vs Price
    sns.regplot(data=df, x="GrLivArea", y="SalePrice", ax=axes[0, 0],
                scatter_kws={"alpha": 0.5, "color": "#2980b9"}, line_kws={"color": "#c0392b"})
    axes[0, 0].set_title("Above Grade Living Area (GrLivArea) vs Sale Price")
    axes[0, 0].set_xlabel("Living Area (sq ft)")
    axes[0, 0].set_ylabel("Sale Price ($)")
    axes[0, 0].yaxis.set_major_formatter('${x:,.0f}')
    
    # 2. TotalBsmtSF vs Price
    sns.regplot(data=df, x="TotalBsmtSF", y="SalePrice", ax=axes[0, 1],
                scatter_kws={"alpha": 0.5, "color": "#16a085"}, line_kws={"color": "#c0392b"})
    axes[0, 1].set_title("Total Basement Area (TotalBsmtSF) vs Sale Price")
    axes[0, 1].set_xlabel("Basement Area (sq ft)")
    axes[0, 1].set_ylabel("Sale Price ($)")
    axes[0, 1].yaxis.set_major_formatter('${x:,.0f}')

    # 3. Amenity Score vs Price
    sns.regplot(data=df, x="amenity_composite_score", y="SalePrice", ax=axes[1, 0],
                scatter_kws={"alpha": 0.5, "color": "#8e44ad"}, line_kws={"color": "#c0392b"})
    axes[1, 0].set_title("Amenity Composite Score vs Sale Price")
    axes[1, 0].set_xlabel("Amenity Score (0 - 100)")
    axes[1, 0].set_ylabel("Sale Price ($)")
    axes[1, 0].yaxis.set_major_formatter('${x:,.0f}')

    # 4. Days on Market vs Price
    sns.regplot(data=df, x="days_on_market", y="SalePrice", ax=axes[1, 1],
                scatter_kws={"alpha": 0.5, "color": "#d35400"}, line_kws={"color": "#c0392b"})
    axes[1, 1].set_title("Days on Market (DOM) vs Sale Price")
    axes[1, 1].set_xlabel("Days on Market (Days)")
    axes[1, 1].set_ylabel("Sale Price ($)")
    axes[1, 1].yaxis.set_major_formatter('${x:,.0f}')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "07_scatter_key_drivers.png"), bbox_inches="tight")
    plt.close(fig)

    # Figure 8: Explicit Outlier Identification (Dean De Cock Recommendations)
    # The Ames dataset contains 2 extreme outliers: GrLivArea > 4000 sq ft but SalePrice < $300,000
    extreme_outliers = df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=df, x="GrLivArea", y="SalePrice", alpha=0.55, color="#34495e", s=40, ax=ax, label="Regular Observations")
    
    # Highlight extreme outliers in bright red
    ax.scatter(extreme_outliers["GrLivArea"], extreme_outliers["SalePrice"],
               color="#e74c3c", s=130, edgecolor="black", zorder=5, label="Dean De Cock Outliers (GrLivArea > 4000 sqft, Price < $300k)")
    
    for _, row in extreme_outliers.iterrows():
        ax.annotate(f"Id {int(row['Id'])}\n(${row['SalePrice']:,.0f}, {row['GrLivArea']} sqft)",
                    xy=(row["GrLivArea"], row["SalePrice"]), xytext=(-60, 25),
                    textcoords="offset points", fontsize=9, fontweight="bold", color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5))

    ax.axvline(4000, color="#e74c3c", linestyle=":", linewidth=1.2)
    ax.set_title("Outlier Diagnostic: Identification of Severe Partial Sales / Abnormal Mansions", pad=15)
    ax.set_xlabel("Above Grade Living Area (GrLivArea sq ft)")
    ax.set_ylabel("Sale Price ($)")
    ax.yaxis.set_major_formatter('${x:,.0f}')
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "08_outliers_grlivarea.png"), bbox_inches="tight")
    plt.close(fig)

    return {
        "identified_extreme_outlier_ids": extreme_outliers["Id"].tolist()
    }


def plot_macro_temporal_dynamics(df: pd.DataFrame):
    """Analyzes the temporal interaction between transaction volumes, median prices, and macro drivers."""
    logger.info("Generating Figure 9: Macroeconomic & Time Trends...")
    
    # Group by YrSold and MoSold
    monthly_stats = df.groupby(["YrSold", "MoSold"]).agg(
        median_price=("SalePrice", "median"),
        avg_rate=("mortgage_rate_pct", "mean"),
        avg_hpi=("local_price_index", "mean"),
        tx_count=("Id", "count")
    ).reset_index()
    
    monthly_stats["Period"] = pd.to_datetime(monthly_stats["YrSold"].astype(str) + "-" + monthly_stats["MoSold"].astype(str) + "-01")
    monthly_stats = monthly_stats.sort_values("Period")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # 1. Monthly Median Price & Transaction Volume
    color1 = "#2c3e50"
    ax1.plot(monthly_stats["Period"], monthly_stats["median_price"], color=color1, marker="o", linewidth=2.2, label="Monthly Median Sale Price")
    ax1.set_ylabel("Median Sale Price ($)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.yaxis.set_major_formatter('${x:,.0f}')
    ax1.set_title("Ames Housing Market Dynamics & Macro Trends (2006 - 2010)", pad=15)

    ax1_vol = ax1.twinx()
    color_vol = "#bdc3c7"
    ax1_vol.bar(monthly_stats["Period"], monthly_stats["tx_count"], width=20, alpha=0.35, color="#3498db", label="Transaction Volume")
    ax1_vol.set_ylabel("Monthly Transactions", color="#2980b9")
    ax1_vol.tick_params(axis="y", labelcolor="#2980b9")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Mortgage Rate vs Local Housing Price Index (HPI)
    color_rate = "#e74c3c"
    ax2.plot(monthly_stats["Period"], monthly_stats["avg_rate"], color=color_rate, linewidth=2, linestyle="-", label="30-Yr Mortgage Rate (%)")
    ax2.set_ylabel("Mortgage Rate (%)", color=color_rate)
    ax2.tick_params(axis="y", labelcolor=color_rate)

    ax2_hpi = ax2.twinx()
    color_hpi = "#27ae60"
    ax2_hpi.plot(monthly_stats["Period"], monthly_stats["avg_hpi"], color=color_hpi, linewidth=2, linestyle="--", label="Local Housing Price Index (HPI)")
    ax2_hpi.set_ylabel("Local HPI (Base 100 = Jan 2006)", color=color_hpi)
    ax2_hpi.tick_params(axis="y", labelcolor=color_hpi)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1_v, labels1_v = ax1_vol.get_legend_handles_labels()
    ax1.legend(lines1 + lines1_v, labels1 + labels1_v, loc="upper right")

    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2_h, labels2_h = ax2_hpi.get_legend_handles_labels()
    ax2.legend(lines2 + lines2_h, labels2 + labels2_h, loc="lower right")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "09_macro_temporal_trends.png"), bbox_inches="tight")
    plt.close(fig)


def calculate_multicollinearity_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Variance Inflation Factor (VIF) for prominent continuous regressors."""
    logger.info("Generating Figure 10: Multicollinearity & VIF Analysis...")
    vif_features = [
        "GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GarageCars",
        "GarageArea", "OverallQual", "YearBuilt", "FullBath", "TotRmsAbvGrd",
        "amenity_composite_score", "days_on_market", "distance_to_school_km"
    ]
    sub_df = df[vif_features].dropna()
    
    # Add intercept column for VIF calculation
    from numpy.linalg import inv
    X = sub_df.values
    # Standardize
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Correlation matrix inverse approximation
    corr_matrix = np.corrcoef(X, rowvar=False)
    inv_corr = inv(corr_matrix)
    vif_values = np.diag(inv_corr)

    vif_df = pd.DataFrame({
        "Feature": vif_features,
        "VIF": np.round(vif_values, 2)
    }).sort_values(by="VIF", ascending=False)

    # Plot VIF horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#c0392b" if v > 5.0 else "#27ae60" for v in vif_df["VIF"]]
    bars = ax.barh(vif_df["Feature"][::-1], vif_df["VIF"][::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
    
    ax.axvline(5.0, color="#e67e22", linestyle="--", linewidth=1.5, label="Moderate Collinearity Threshold (VIF = 5.0)")
    ax.axvline(10.0, color="#c0392b", linestyle=":", linewidth=1.5, label="Severe Multicollinearity Threshold (VIF = 10.0)")
    
    ax.set_title("Variance Inflation Factor (VIF) for Primary Regressors\n(Red: VIF > 5 indicates redundant dimensionality)", pad=15)
    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.legend(loc="lower right")

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.1f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "10_multicollinearity_vif.png"), bbox_inches="tight")
    plt.close(fig)

    return vif_df


def main():
    logger.info("=" * 60)
    logger.info("STARTING MODULE 2: EXPLORATORY DATA ANALYSIS PIPELINE")
    logger.info("=" * 60)
    
    ensure_dirs()
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Input file not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

    summary_metrics = {}

    # 1. Target Distribution
    summary_metrics["target_distribution"] = plot_target_distribution(df)

    # 2. Correlations
    summary_metrics["correlation_analysis"] = plot_correlations(df)

    # 3. Spatial & Neighborhood Patterns
    plot_neighborhood_patterns(df)

    # 4. Property Typology, Age & Renovation
    plot_structural_and_vintage_patterns(df)

    # 5. Key Bivariate Drivers & Outliers
    summary_metrics["outlier_diagnostics"] = plot_key_drivers_and_outliers(df)

    # 6. Macro Trends
    plot_macro_temporal_dynamics(df)

    # 7. Multicollinearity VIF
    vif_df = calculate_multicollinearity_vif(df)
    summary_metrics["vif_table"] = vif_df.to_dict(orient="records")

    # Export metrics JSON
    with open("reports/eda_summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4)

    logger.info("Exported summary metrics to reports/eda_summary_metrics.json")
    logger.info("=" * 60)
    logger.info("MODULE 2 EDA PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"All 10 figures exported to: {os.path.abspath(OUTPUT_DIR)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
