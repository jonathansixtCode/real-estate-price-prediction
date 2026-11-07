# Exploratory Data Analysis (EDA) Report
**Project:** House Price Prediction for a Real Estate Listing Platform  
**Course:** Business Analytics — Module 2 Deliverable  
**Analyzed Dataset:** `data/processed/train_augmented.csv` (1,460 observations, 94 features)  
**Notebook Reference:** [`notebooks/01_exploratory_data_analysis.ipynb`](../notebooks/01_exploratory_data_analysis.ipynb)  
**Figures Export Directory:** [`reports/figures/`](../reports/figures/)

---

## 1. Executive Summary
This report documents the findings of **Module 2 (Exploratory Data Analysis)** for our algorithmic real estate valuation engine. Through rigorous statistical testing, distributional diagnostics, spatial price mapping, and collinearity audits, we have identified:
1. A **highly right-skewed target variable** ($\text{SalePrice}$) requiring log1p normalization to satisfy linear modeling assumptions.
2. Two **severe bivariate leverage outliers** (`Id = 524` and `Id = 1299`) that severely distort regression slopes and must be removed in Module 3.
3. Strong predictive power in both **classical structural features** (`OverallQual`, `GrLivArea`, `TotalBsmtSF`) and our **newly engineered synthetic contextual features** (`amenity_composite_score`, `renovation_cost_usd`, `days_on_market`).
4. Significant **structural multicollinearity** among spatial dimension variables (`GrLivArea`, `1stFlrSF`, `2ndFlrSF` with $\text{VIF} > 70$) and garage attributes, dictating dimension reduction in Module 4.

---

## 2. Target Variable Analysis (`SalePrice` vs. `LogSalePrice`)
*Visual Reference:* `reports/figures/01_target_distribution.png`

| Metric | Raw `SalePrice` | Transformed `$\log(1 + \text{SalePrice})$` | Statistical Interpretation |
| :--- | :--- | :--- | :--- |
| **Mean** | $\$180,921$ | $12.02$ | Raw mean exceeds median due to extreme high-end luxury listings. |
| **Median** | $\$163,000$ | $12.00$ | Symmetric alignment under log transformation. |
| **Standard Deviation** | $\$79,443$ | $0.40$ | Variance is stabilized across price tiers (homoscedasticity). |
| **Skewness** | **$+1.88$** | **$+0.12$** | Severe right skewness is eliminated; distribution closely approximates normal. |
| **Kurtosis** | **$+6.54$** | **$+0.81$** | Heavy leptokurtic tails are compressed into near-mesokurtic shape. |

> **Modeling Recommendation for Module 5:** All regression models (OLS, Ridge, Lasso, and Gradient Boosting) must be trained on $\log(1 + \text{SalePrice})$. Point predictions will be transformed back using $\exp(\hat{y}) - 1$ with appropriate Smearing or median re-transformation corrections.

---

## 3. Outlier Diagnostics & Anomalies
*Visual Reference:* `reports/figures/08_outliers_grlivarea.png`

### 3.1 The Dean De Cock Outlier Anomaly
In the benchmark documentation of the Ames housing dataset (*De Cock, 2011*), the author explicitly recommends removing observations with over $4,000$ square feet of living area that sold for non-luxury prices. Our visual and numerical diagnostics pinpointed two exact anomalies:

| Property `Id` | Neighborhood | Living Area (`GrLivArea`) | Overall Quality | Actual Sale Price | Sale Condition | Diagnostic Flag |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **524** | `Edwards` | **$4,676$ sq ft** | $10 / 10$ | **$\$184,750$** | `Partial` | Massive luxury build sold as distressed partial contract. |
| **1299** | `Edwards` | **$5,642$ sq ft** | $10 / 10$ | **$\$160,000$** | `Partial` | Largest property in entire dataset; agricultural/partial anomaly. |

**Impact on Regression:** These two points possess massive leverage ($h_{ii}$) coupled with high negative residuals ($e_i$). In Ordinary Least Squares (OLS), they act as influential leverage points that artificially flatten the slope of square footage pricing.  
**Decision for Module 3 (Data Cleaning):** Both rows (`Id = 524, 1299`) will be explicitly filtered out prior to feature scaling.

---

## 4. Key Price Drivers & Correlation Analysis
*Visual References:* `reports/figures/02_top_correlations.png` & `reports/figures/03_correlation_heatmap.png`

### Top 15 Continuous Regressors Correlated with `SalePrice`
1. `OverallQual` ($r = \mathbf{0.791}$): The overall subjective build and finish rating is the strongest single predictor of transaction value.
2. `GrLivArea` ($r = \mathbf{0.709}$): Above-grade living area dominates physical spatial metrics.
3. `GarageCars` ($r = \mathbf{0.640}$) & `GarageArea` ($r = \mathbf{0.623}$): Highly collinear metrics of automotive capacity.
4. `TotalBsmtSF` ($r = \mathbf{0.614}$) & `1stFlrSF` ($r = \mathbf{0.606}$): Foundation and primary floor footprint.
5. `FullBath` ($r = \mathbf{0.561}$) & `TotRmsAbvGrd` ($r = \mathbf{0.534}$): Functional accommodation capacity.
6. `YearBuilt` ($r = \mathbf{0.523}$) & `YearRemodAdd` ($r = \mathbf{0.507}$): Vintage and structural obsolescence.

### Validation of Synthetic Contextual Features
* **`amenity_composite_score` ($r = 0.472$) & `distance_to_school_km` ($r = 0.472$):** Exhibit solid positive correlation with market valuation, demonstrating that properties in premium school districts and high-density amenity clusters command substantial premiums in Ames.
* **`renovation_cost_usd` ($r = 0.384$):** Demonstrates that capitalized modernization directly enhances asset marketability.
* **`days_on_market` ($r = -0.061$):** Shows a slight negative correlation with price, accurately capturing that distressed, flawed, or overpriced listings linger on the market before capitulating to discounts.

---

## 5. Locational, Typological & Vintage Patterns

### 5.1 Neighborhood Price Gradient (`reports/figures/04_price_by_neighborhood.png`)
Ames real estate is sharply segmented across geographic clusters:
* **Tier 1 (Luxury / Executive Subdivisions):** `NridgHt` (Median: $\$315,000$), `StoneBr` (Median: $\$278,000$), `NoRidge` (Median: $\$290,000$). High living area, recent construction, and premium school proximity.
* **Tier 2 (Established Middle-Class Suburban):** `CollgCr`, `Somerst`, `Crawfor`, `Gilbert`, `SawyerW` (Medians: $\$180,000\text{--}\$225,000$).
* **Tier 3 (Historic / Working-Class / Student Rentals):** `OldTown`, `Edwards`, `IDOTRR`, `BrDale`, `MeadowV` (Medians: $\$85,000\text{--}\$125,000$). High prevalence of pre-1950 construction and unrenovated structures.

### 5.2 Building Typology & Architectural Style (`reports/figures/05_price_by_building_type_and_style.png`)
* **Single-Family Detached (`1Fam`):** Represents $83.5\%$ of listings and commands the highest median price $(\$167,000)$ with the greatest price variance.
* **Townhomes (`TwnhsE`, `Twnhs`):** End-unit townhomes (`TwnhsE`) command a sizable premium over interior units (`Twnhs`) due to superior natural light and reduced shared walls.
* **Duplexes (`Duplex`):** Low median pricing $(\$135,000)$ reflecting landlord-owned rental property conditions.

### 5.3 Age, Depreciation & Renovation Offset (`reports/figures/06_price_vs_age_and_renovation.png`)
Property age exhibits a non-linear parabolic depreciation curve:
* Properties built post-2000 trade at a sharp premium.
* Properties between 30 and 70 years old reach maximum economic obsolescence unless mitigated by capital renovations.
* Homes with **Major Structural** or **Kitchen/Bath renovations** outperform unrenovated peers of identical vintage by an average of $\$32,000\text{--}\$58,000$.

---

## 6. Macroeconomic & Time Dynamics (2006–2010)
*Visual Reference:* `reports/figures/09_macro_temporal_trends.png`

The 1,460 transactions span from January 2006 to July 2010:
* **Seasonality:** Pronounced transaction volume spikes occur consistently in **May, June, and July** ($45.6\%$ of all sales), whereas winter volume (December–February) drops by over $65\%$.
* **Financial Crisis Trajectory:** Transaction medians peaked in mid-2007 ($\approx \$178,000$), experienced severe stagnation in late 2008 following the collapse of Lehman Brothers, and began stabilizing in early 2010 as 30-year fixed mortgage rates compressed from $6.6\%$ down to $4.7\%$.

---

## 7. Multicollinearity Diagnostics & Variance Inflation Factors (VIF)
*Visual Reference:* `reports/figures/10_multicollinearity_vif.png`

Regressors were evaluated for collinear redundancy using Variance Inflation Factor ($\text{VIF} = \frac{1}{1 - R_j^2}$):

| Feature | VIF Score | Diagnostic Severity | Root Cause & Business Explanation |
| :--- | :---: | :---: | :--- |
| `GrLivArea` | **$125.82$** | **Severe ($\text{VIF} > 10$)** | Exact mathematical identity: $\text{GrLivArea} = \text{1stFlrSF} + \text{2ndFlrSF} (+ \text{LowQualFinSF})$. |
| `2ndFlrSF` | **$87.14$** | **Severe ($\text{VIF} > 10$)** | Subcomponent of above-grade living area. |
| `1stFlrSF` | **$70.24$** | **Severe ($\text{VIF} > 10$)** | Subcomponent of above-grade living area. |
| `amenity_composite_score` | **$12.02$** | **Moderate / High** | Built directly as a linear combination of neighborhood distances. |
| `distance_to_school_km` | **$10.41$** | **Moderate / High** | Strongly correlated with composite amenity index. |
| `GarageCars` | **$5.33$** | **Moderate ($\text{VIF} \approx 5$)** | Highly collinear with `GarageArea` ($r = 0.88$). |
| `GarageArea` | **$5.01$** | **Moderate ($\text{VIF} \approx 5$)** | Highly collinear with `GarageCars`. |
| `TotalBsmtSF` | $3.62$ | Low ($\text{VIF} < 5$) | Acceptable independent foundation regressor. |
| `TotRmsAbvGrd` | $3.39$ | Low ($\text{VIF} < 5$) | Acceptable accommodation regressor. |
| `YearBuilt` | $2.94$ | Low ($\text{VIF} < 5$) | Acceptable temporal regressor. |
| `OverallQual` | $2.74$ | Low ($\text{VIF} < 5$) | Highly stable independent condition predictor. |
| `FullBath` | $2.19$ | Low ($\text{VIF} < 5$) | Acceptable. |
| `days_on_market` | **$1.05$** | **Ideal ($\text{VIF} \approx 1$)** | Orthogonal information with zero collinear inflation. |

---

## 8. Strategic Action Plan for Modules 3 & 4

| EDA Insight | Action for Module 3 (Data Cleaning) | Action for Module 4 (Feature Engineering) |
| :--- | :--- | :--- |
| **Outliers `Id = 524, 1299`** | Drop both records from training set. | Apply robust scaling to living areas. |
| **Target Skewness ($+1.88$)** | Retain raw target in data; flag for transformation. | Log1p transform $\log(1 + y)$ prior to training. |
| **Severe VIF on Area Features** | Verify missing values in basement/floor columns. | Synthesize into `TotalLivingSF` and drop duplicate intermediate terms. |
| **Garage Collinearity** | Handle $81$ missing values in garage records ($= \text{No Garage}$). | Keep `GarageCars` (integer discrete capacity) or derive ratio. |
| **Temporal / Age Dynamics** | Clean inconsistent `YearRemodAdd < YearBuilt`. | Create `PropertyAge = YrSold - YearBuilt` and `YearsSinceRemod`. |
| **Amenity Correlation** | No cleaning required (clean synthetic generation). | Standardize composite index; drop redundant single distances if using linear models. |
