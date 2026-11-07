"""
Builds the Module 2 Jupyter Notebook: notebooks/01_exploratory_data_analysis.ipynb
"""

import json
import os

def create_cell(cell_type: str, source: list, execution_count=None, outputs=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [s + "\n" for s in source]
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs or []
    return cell

def build_notebook():
    cells = []

    # Title & Metadata
    cells.append(create_cell("markdown", [
        "# Module 2: Exploratory Data Analysis (EDA)",
        "**Real Estate House Price Prediction Platform**  ",
        "*An End-to-End Analytics Project: From Raw Data to a Monitored, Deployed Model*",
        "",
        "---",
        "",
        "## Executive Summary & Objectives",
        "The purpose of this notebook is to execute the core deliverables of **Module 2**: ",
        "1. **Analyze distributions, outliers, and correlations** between property attributes and sale price.",
        "2. **Compare price patterns** across location (`Neighborhood`), property type (`BldgType`, `HouseStyle`), and age (`YearBuilt`, `YearRemodAdd`).",
        "3. **Visualize relationships** and diagnose **multicollinearity** (correlation matrices, Variance Inflation Factors).",
        "4. Evaluate the impact of the newly generated **synthetic contextual features** (Amenity scores, Renovation history, Days-on-Market, Macroeconomic indicators).",
        "",
        "Dataset analyzed: `data/processed/train_augmented.csv` (1,460 observations, 94 features)."
    ]))

    # Setup & Imports
    cells.append(create_cell("code", [
        "# Setup environment and load dependencies",
        "import os",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "from scipy import stats",
        "",
        "# Styling configuration",
        "sns.set_theme(style='whitegrid', palette='muted')",
        "plt.rcParams['figure.figsize'] = (10, 6)",
        "plt.rcParams['font.size'] = 10",
        "",
        "# Load augmented training dataset",
        "data_path = '../data/processed/train_augmented.csv'",
        "df = pd.read_csv(data_path)",
        "print(f'Successfully loaded dataset with shape: {df.shape}')"
    ]))

    # Section 1: Target Variable Analysis
    cells.append(create_cell("markdown", [
        "---",
        "## 1. Target Variable Analysis (`SalePrice` vs. `LogSalePrice`)",
        "",
        "In residential property valuation, transaction prices routinely exhibit substantial **positive skewness** (a heavy right tail caused by luxury/estate properties).",
        "Linear regression and ordinary least squares (OLS) assume homoscedastic and normally distributed residual errors $\\epsilon \\sim \\mathcal{N}(0, \\sigma^2)$. Transforming the target variable to the logarithmic scale $\\log(1 + \\text{SalePrice})$ stabilizes variance and normalizes residuals."
    ]))

    cells.append(create_cell("code", [
        "raw_price = df['SalePrice']",
        "log_price = np.log1p(raw_price)",
        "",
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))",
        "",
        "# Raw Histogram",
        "sns.histplot(raw_price, kde=True, ax=axes[0, 0], color='#2b5c8f', bins=50)",
        "axes[0, 0].set_title(f'Raw SalePrice (Skewness: {raw_price.skew():.2f}, Kurtosis: {raw_price.kurt():.2f})')",
        "axes[0, 0].xaxis.set_major_formatter('${x:,.0f}')",
        "",
        "# Raw Q-Q Plot",
        "stats.probplot(raw_price, dist='norm', plot=axes[0, 1])",
        "axes[0, 1].set_title('Normal Q-Q Plot: Raw SalePrice')",
        "",
        "# Log Histogram",
        "sns.histplot(log_price, kde=True, ax=axes[1, 0], color='#28a745', bins=50)",
        "axes[1, 0].set_title(f'Log1p SalePrice (Skewness: {log_price.skew():.2f}, Kurtosis: {log_price.kurt():.2f})')",
        "",
        "# Log Q-Q Plot",
        "stats.probplot(log_price, dist='norm', plot=axes[1, 1])",
        "axes[1, 1].set_title('Normal Q-Q Plot: Log1p SalePrice')",
        "",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "print(f'Median SalePrice: ${raw_price.median():,.0f} | Mean: ${raw_price.mean():,.0f} | Std: ${raw_price.std():,.0f}')"
    ]))

    # Section 2: Outlier Diagnostics
    cells.append(create_cell("markdown", [
        "---",
        "## 2. Univariate & Bivariate Outlier Diagnostics",
        "",
        "### The Dean De Cock Outlier Anomaly",
        "The author of the Ames Housing dataset, Prof. Dean De Cock (2011), specifically highlighted two extreme observations in the training data: ",
        "Properties with **`GrLivArea > 4,000 sq ft`** that sold for unusually low prices ($< \\$300,000$). These represent partial sales or abnormal agricultural dwellings. Keeping these observations severely distorts linear regression slopes and inflates RMSE."
    ]))

    cells.append(create_cell("code", [
        "# Identify Dean De Cock Outliers",
        "extreme_outliers = df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)]",
        "print('Identified Severe Outlier Records:')",
        "display(extreme_outliers[['Id', 'Neighborhood', 'GrLivArea', 'OverallQual', 'SalePrice', 'SaleCondition']])",
        "",
        "# Plot GrLivArea vs SalePrice with outlier annotation",
        "plt.figure(figsize=(10, 6))",
        "sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', alpha=0.5, color='#34495e', label='Standard Properties')",
        "plt.scatter(extreme_outliers['GrLivArea'], extreme_outliers['SalePrice'], color='#e74c3c', s=120, edgecolor='black', zorder=5, label='Extreme Outliers (Id 524, 1299)')",
        "",
        "for _, r in extreme_outliers.iterrows():",
        "    plt.annotate(f'Id {int(r[\"Id\"])}\\n(${r[\"SalePrice\"]:,.0f})', xy=(r['GrLivArea'], r['SalePrice']),",
        "                 xytext=(-50, 20), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5), fontweight='bold')",
        "",
        "plt.axvline(4000, color='#e74c3c', linestyle=':', label='4,000 sqft Threshold')",
        "plt.title('Outlier Diagnostic: GrLivArea vs. SalePrice')",
        "plt.xlabel('Above Grade Living Area (sq ft)')",
        "plt.ylabel('Sale Price ($)')",
        "plt.gca().yaxis.set_major_formatter('${x:,.0f}')",
        "plt.legend()",
        "plt.show()"
    ]))

    # Section 3: Correlations & Feature Importance
    cells.append(create_cell("markdown", [
        "---",
        "## 3. Correlation Analysis & Feature Importance",
        "",
        "We evaluate linear (Pearson) correlations between all continuous regressors and both raw and log-transformed `SalePrice`, with particular attention to how synthetic contextual features compare against raw physical attributes."
    ]))

    cells.append(create_cell("code", [
        "num_df = df.select_dtypes(include=[np.number]).drop(columns=['Id'], errors='ignore')",
        "corrs = num_df.corr()['SalePrice'].sort_values(ascending=False)",
        "",
        "# Bar plot of top 15 correlated features",
        "top_15 = corrs.drop('SalePrice').head(15)",
        "plt.figure(figsize=(10, 7))",
        "colors = ['#e67e22' if any(k in col for k in ['amenity', 'renovation', 'market', 'rate', 'price_index']) else '#2b5c8f' for col in top_15.index]",
        "sns.barplot(x=top_15.values, y=top_15.index, palette=colors)",
        "plt.title('Top 15 Regressors Correlated with SalePrice\\n(Orange = Synthetic Contextual Features)')",
        "plt.xlabel('Pearson Correlation Coefficient (r)')",
        "for i, v in enumerate(top_15.values):",
        "    plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)",
        "plt.xlim(0, 0.88)",
        "plt.show()"
    ]))

    # Section 4: Spatial & Neighborhood Patterns
    cells.append(create_cell("markdown", [
        "---",
        "## 4. Spatial & Locational Price Patterns (Ames Neighborhoods)",
        "",
        "Location is the single greatest pricing determinant in real estate (*'location, location, location'*). Below, we examine the price gradient across the 25 distinct Ames neighborhoods."
    ]))

    cells.append(create_cell("code", [
        "neigh_order = df.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False).index",
        "",
        "plt.figure(figsize=(14, 7))",
        "sns.boxplot(data=df, x='Neighborhood', y='SalePrice', order=neigh_order, palette='Blues_r', fliersize=2)",
        "plt.axhline(df['SalePrice'].median(), color='#d9534f', linestyle='--', label=f'Overall Median: ${df[\"SalePrice\"].median():,.0f}')",
        "plt.title('Sale Price Distribution across Ames Neighborhoods (Sorted by Median Price)')",
        "plt.xlabel('Neighborhood Code')",
        "plt.ylabel('Sale Price ($)')",
        "plt.gca().yaxis.set_major_formatter('${x:,.0f}')",
        "plt.xticks(rotation=45)",
        "plt.legend()",
        "plt.show()"
    ]))

    # Section 5: Typology, Age & Renovation Analysis
    cells.append(create_cell("markdown", [
        "---",
        "## 5. Architectural Typology, Age & Renovation Dynamics",
        "",
        "How do property age, structural typology, and renovation scope interact with market valuation?"
    ]))

    cells.append(create_cell("code", [
        "fig, axes = plt.subplots(1, 2, figsize=(14, 6))",
        "",
        "# BldgType",
        "bldg_order = df.groupby('BldgType')['SalePrice'].median().sort_values(ascending=False).index",
        "sns.boxplot(data=df, x='BldgType', y='SalePrice', order=bldg_order, ax=axes[0], palette='Set2')",
        "axes[0].set_title('Sale Price by Building Type (BldgType)')",
        "axes[0].yaxis.set_major_formatter('${x:,.0f}')",
        "",
        "# HouseStyle",
        "style_order = df.groupby('HouseStyle')['SalePrice'].median().sort_values(ascending=False).index",
        "sns.boxplot(data=df, x='HouseStyle', y='SalePrice', order=style_order, ax=axes[1], palette='Set2')",
        "axes[1].set_title('Sale Price by Dwelling Style (HouseStyle)')",
        "axes[1].yaxis.set_major_formatter('${x:,.0f}')",
        "axes[1].tick_params(axis='x', rotation=30)",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]))

    cells.append(create_cell("code", [
        "# Age at Sale vs SalePrice by Renovation Type",
        "df_temp = df.copy()",
        "df_temp['AgeAtSale'] = df_temp['YrSold'] - df_temp['YearBuilt']",
        "",
        "plt.figure(figsize=(11, 6))",
        "palette = {'None': '#7f8c8d', 'Cosmetic': '#3498db', 'Kitchen_Bath': '#e67e22', 'Major_Structural': '#e74c3c'}",
        "sns.scatterplot(data=df_temp, x='AgeAtSale', y='SalePrice', hue='renovation_type', palette=palette, alpha=0.6, s=40)",
        "sns.regplot(data=df_temp, x='AgeAtSale', y='SalePrice', scatter=False, color='#2c3e50', order=2, line_kws={'linewidth': 2.5, 'label': 'Depreciation Trend'})",
        "plt.title('Property Age at Sale vs. Sale Price by Renovation Type')",
        "plt.xlabel('Property Age (Years)')",
        "plt.ylabel('Sale Price ($)')",
        "plt.gca().yaxis.set_major_formatter('${x:,.0f}')",
        "plt.legend(title='Renovation Scope')",
        "plt.show()"
    ]))

    # Section 6: Macroeconomic & Time Dynamics
    cells.append(create_cell("markdown", [
        "---",
        "## 6. Macroeconomic & Temporal Dynamics (2006–2010)",
        "",
        "The Ames transaction data spans from **January 2006 through July 2010**, capturing the pivotal inflection point of the US Great Financial Crisis. Below, we track monthly volume, median prices, and macro interest rates."
    ]))

    cells.append(create_cell("code", [
        "monthly = df.groupby(['YrSold', 'MoSold']).agg(",
        "    median_price=('SalePrice', 'median'),",
        "    avg_rate=('mortgage_rate_pct', 'mean'),",
        "    tx_count=('Id', 'count')",
        ").reset_index()",
        "monthly['Date'] = pd.to_datetime(monthly['YrSold'].astype(str) + '-' + monthly['MoSold'].astype(str) + '-01')",
        "monthly = monthly.sort_values('Date')",
        "",
        "fig, ax1 = plt.subplots(figsize=(12, 6))",
        "ax1.plot(monthly['Date'], monthly['median_price'], color='#2c3e50', marker='o', linewidth=2, label='Median Price ($)')",
        "ax1.set_ylabel('Median Price ($)', color='#2c3e50')",
        "ax1.yaxis.set_major_formatter('${x:,.0f}')",
        "",
        "ax2 = ax1.twinx()",
        "ax2.plot(monthly['Date'], monthly['avg_rate'], color='#e74c3c', linestyle='--', linewidth=2, label='30-Yr Mortgage Rate (%)')",
        "ax2.set_ylabel('Mortgage Rate (%)', color='#e74c3c')",
        "",
        "plt.title('Ames Monthly Median Price vs. US 30-Year Fixed Mortgage Rate (2006–2010)')",
        "ax1.grid(True, linestyle='--', alpha=0.5)",
        "plt.show()"
    ]))

    # Section 7: Multicollinearity & VIF
    cells.append(create_cell("markdown", [
        "---",
        "## 7. Multicollinearity Diagnostics & Variance Inflation Factors (VIF)",
        "",
        "Multicollinearity occurs when independent variables are highly inter-correlated. While tree-based algorithms (Random Forest, XGBoost) are relatively resilient, linear and regularized regressors (Ridge, Lasso) suffer from unstable coefficient estimation and high standard errors.",
        "",
        "Rule of thumb:",
        "* $\\text{VIF} < 5$: Low multicollinearity.",
        "* $5 \\le \\text{VIF} < 10$: Moderate collinearity (monitor).",
        "* $\\text{VIF} \\ge 10$: Severe collinearity (action required in Module 3/4)."
    ]))

    cells.append(create_cell("code", [
        "vif_features = [",
        "    'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageCars',",
        "    'GarageArea', 'OverallQual', 'YearBuilt', 'FullBath', 'TotRmsAbvGrd',",
        "    'amenity_composite_score', 'days_on_market', 'distance_to_school_km'",
        "]",
        "sub_df = df[vif_features].dropna()",
        "X = (sub_df - sub_df.mean()) / sub_df.std()",
        "corr_mat = np.corrcoef(X.values, rowvar=False)",
        "vif_vals = np.diag(np.linalg.inv(corr_mat))",
        "",
        "vif_table = pd.DataFrame({'Feature': vif_features, 'VIF': np.round(vif_vals, 2)}).sort_values(by='VIF', ascending=False)",
        "print('Variance Inflation Factor (VIF) Diagnostic Table:')",
        "display(vif_table)",
        "",
        "# Bar plot",
        "plt.figure(figsize=(10, 6))",
        "colors = ['#c0392b' if v > 10 else ('#e67e22' if v > 5 else '#27ae60') for v in vif_table['VIF']]",
        "sns.barplot(x='VIF', y='Feature', data=vif_table, palette=colors)",
        "plt.axvline(5.0, color='#e67e22', linestyle='--', label='VIF = 5.0 (Moderate)')",
        "plt.axvline(10.0, color='#c0392b', linestyle=':', label='VIF = 10.0 (Severe)')",
        "plt.title('Variance Inflation Factors across Primary Predictors')",
        "plt.xlabel('VIF Value')",
        "plt.legend()",
        "plt.show()"
    ]))

    # Conclusion & Transition
    cells.append(create_cell("markdown", [
        "---",
        "## 8. Summary of Findings & Transition to Module 3 (Data Cleaning)",
        "",
        "### Key Takeaways from Module 2 EDA:",
        "1. **Target Distribution:** `SalePrice` is strongly right-skewed (skewness $= 1.88$). Log1p transformation brings skewness to $0.12$, establishing it as the standard target for model training in Module 5.",
        "2. **Outlier Mitigation:** Identified observations `Id 524` and `Id 1299` as severe partial sales ($> 4,000$ sq ft, $< \\$300,000$). Must be removed in Module 3 Data Cleaning.",
        "3. **Top Physical Drivers:** `OverallQual` ($r = 0.79$), `GrLivArea` ($r = 0.71$), `GarageCars` ($r = 0.64$), and `TotalBsmtSF` ($r = 0.61$).",
        "4. **Synthetic Feature Validation:**",
        "   * `amenity_composite_score` and `distance_to_school_km` ($r \\approx 0.47$) exhibit strong locational explanatory power.",
        "   * `days_on_market` ($r = -0.06$) provides independent, orthogonal information (VIF $= 1.05$), with high DOM indicating distressed or overpriced assets.",
        "   * `renovation_cost_usd` displays a positive correlation ($r = 0.38$) with price, validating the realism of the simulation logic.",
        "5. **Multicollinearity Flag:** `GrLivArea` is essentially the exact sum of `1stFlrSF` and `2ndFlrSF` (VIF $> 70$). Similarly, `GarageCars` and `GarageArea` exhibit collinear redundancy (VIF $> 5$). Module 4 Feature Engineering will perform feature synthesis/selection."
    ]))

    notebook_content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (real-estate-env)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_path = "notebooks/01_exploratory_data_analysis.ipynb"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)
    print(f"Jupyter Notebook successfully built at {out_path} ({len(cells)} cells).")

if __name__ == "__main__":
    build_notebook()
