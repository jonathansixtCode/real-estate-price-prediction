# Module 1: Business Understanding & Synthetic Contextual Data Generation
**Project:** House Price Prediction for a Real Estate Listing Platform (Ames, Iowa)

---

## 1. Executive Summary & Objectives
The goal of Module 1 is to establish the business framing for an automated real estate valuation model in Ames, Iowa, and enrich the foundational Kaggle Ames Housing dataset with realistic, deterministic contextual data (`seed=42`). 

### Core Objectives:
1. **Business Framing:** Define the core property valuation problem, account for asymmetric error costs (e.g., overvaluation vs. undervaluation risks for buyers, sellers, and platform trust), and establish target Key Performance Indicators (KPIs).
2. **Base Data Ingestion:** Load the raw Kaggle `train.csv` ($1,460$ rows) and `test.csv` ($1,459$ rows) datasets.
3. **Synthetic Feature Augmentation:** Generate **13 deterministic synthetic contextual features** spanning neighborhood amenities, renovation history, market liquidity, and macroeconomic conditions.
4. **Data Validation & Sanity Checks:** Verify data completeness, statistical distributions, logical ranges, and real-world business realism across both datasets.

---

## 2. Dataset Overview & Pipeline Execution
Running the deterministic pipeline (`run_module_1(..., seed=42)`) successfully augmented the baseline data with zero missing values across all newly generated features:

| Dataset | Output File | Rows | Total Columns | Missing Synthetic Values |
| :--- | :--- | :---: | :---: | :---: |
| **Training Set** | `train_augmented.csv` | $1,460$ | $94$ | $0$ |
| **Test Set** | `test_augmented.csv` | $1,459$ | $93$ | $0$ |

---

## 3. Synthetic Contextual Features (13 Features)
The 13 engineered features capture external market, location, and property-history drivers that are absent from standard structural housing records:

### A. Location & Neighborhood Amenities
* `distance_to_school_km`: Proximity to the nearest school (in kilometers).
* `distance_to_hospital_km`: Proximity to the nearest hospital/medical facility (in kilometers).
* `distance_to_transit_km`: Proximity to the nearest public transit stop (in kilometers).
* `amenity_composite_score`: Aggregated location desirability index ($0$–$100$ scale) reflecting overall proximity to key infrastructure.

### B. Property Renovation History
* `has_renovation`: Binary flag indicating whether the property underwent documented renovation ($1 = \text{Yes}$, $0 = \text{No}$).
* `renovation_type`: Categorical classification of the remodel scope (e.g., `None`, `Cosmetic`, `Kitchen_Bath`, etc.).
* `renovation_cost_usd`: Estimated capital expenditure in USD ($\$0.00$ for non-renovated homes).
* `renovation_contractor`: Name of the licensed contractor executing the renovation.
* `renovation_permit_id`: Official municipal permit identifier (formatted as `PRM-YYYY-XXXXX`).

### C. Market Liquidity & Macroeconomic Indicators
* `days_on_market`: Number of days the listing remained active prior to sale (DOM).
* `mortgage_rate_pct`: Prevailing 30-year fixed mortgage interest rate (%) at the time of sale.
* `local_price_index`: Regional Ames housing price index tracking local market momentum.
* `unemployment_rate_pct`: Local unemployment rate (%) reflecting regional economic health.

---

## 4. Summary Statistics & Validation Results

### Key Validation Metrics
* **Training Set (`train_augmented`):**
  * **Renovation Rate:** $54.45\%$
  * **Average Days on Market (DOM):** $40.50\text{ days}$
  * **Average Amenity Composite Score:** $60.82$
* **Test Set (`test_augmented`):**
  * **Renovation Rate:** $52.02\%$
  * **Average Days on Market (DOM):** $39.93\text{ days}$
  * **Average Amenity Composite Score:** $60.73$

### Numerical Feature Distribution (`train_augmented`, $N = 1,460$)

| Feature | Mean | Std. Dev. | Min | 25% (Q1) | Median (50%) | 75% (Q3) | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `distance_to_school_km` | $1.33$ | $0.63$ | $0.31$ | $0.82$ | $1.23$ | $1.73$ | $3.44$ |
| `distance_to_hospital_km` | $3.35$ | $1.40$ | $0.52$ | $2.26$ | $3.52$ | $4.46$ | $6.57$ |
| `distance_to_transit_km` | $0.56$ | $0.31$ | $0.08$ | $0.33$ | $0.45$ | $0.76$ | $1.68$ |
| `amenity_composite_score` | $60.82$ | $17.13$ | $25.00$ | $48.19$ | $63.78$ | $74.55$ | $91.43$ |
| `has_renovation` | $0.54$ | $0.50$ | $0.00$ | $0.00$ | $1.00$ | $1.00$ | $1.00$ |
| `renovation_cost_usd` | $\$10,764.22$ | $\$16,633.16$ | $\$0.00$ | $\$0.00$ | $\$6,297.72$ | $\$13,017.09$ | $\$125,000.00$ |
| `days_on_market` | $40.50$ | $21.13$ | $9.00$ | $26.00$ | $36.00$ | $50.00$ | $209.00$ |
| `mortgage_rate_pct` | $5.78\%$ | $0.67\%$ | $4.51\%$ | $5.04\%$ | $6.06\%$ | $6.35\%$ | $6.56\%$ |
| `local_price_index` | $98.93$ | $4.63$ | $90.76$ | $93.54$ | $99.95$ | $103.15$ | $105.11$ |
| `unemployment_rate_pct` | $4.66\%$ | $1.06\%$ | $3.40\%$ | $3.77\%$ | $4.14\%$ | $5.99\%$ | $6.44\%$ |

---

## 5. Visual Realism Verification
Visual inspection via distribution plots and boxplots confirms realistic domain behavior:
1. **Amenity Composite Score:** Exhibits a well-bounded, slightly left-skewed distribution ($25.0$ to $91.43$) peaking around $64$–$75$, reflecting typical suburban neighborhood clustering.
2. **Renovation Cost (Renovated Homes):** Right-skewed cost distribution among the $54.45\%$ renovated properties, capturing frequent minor/cosmetic updates alongside a long tail of major structural/kitchen remodels up to $\$125,000$.
3. **Days on Market (DOM):** Right-skewed liquidity distribution centered around a median of $36\text{ days}$ ($\text{IQR}: 26\text{–}50\text{ days}$) with realistic slow-moving outlier listings extending up to $209\text{ days}$.
4. **30-Year Mortgage Rate by Sale Year (`YrSold`):** Accurately reflects temporal macroeconomic shifts across sale years ($2006\text{–}2010$), showing higher rates ($\approx 6.0\%\text{–}6.5\%$) in earlier years and a downward trend toward $\approx 4.5\%\text{–}5.0\%$ in later years.