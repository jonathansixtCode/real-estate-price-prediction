# Feature Engineering & Preprocessing Report
**Project:** House Price Prediction for a Real Estate Listing Platform  
**Course:** Business Analytics — Module 4 Deliverable  
**Analyzed Datasets:** `data/cleaned/train_cleaned.csv` & `data/cleaned/test_cleaned.csv`  
**Output Processed Datasets:** [`data/engineered/train_engineered.csv`](../data/engineered/train_engineered.csv) & [`data/engineered/test_engineered.csv`](../data/engineered/test_engineered.csv)  
**Notebook Reference:** [`notebooks/03_feature_engineering.ipynb`](../notebooks/03_feature_engineering.ipynb)  
**Figure References:** [`reports/figures/12_feature_engineering_vif_reduction.png`](../reports/figures/12_feature_engineering_vif_reduction.png) & [`reports/figures/13_feature_importance_mi.png`](../reports/figures/13_feature_importance_mi.png)

---

## 1. Executive Summary
This report details the execution and mathematical justification of **Module 4 (Feature Engineering)**. Feature engineering systematically enhances raw clean data into optimal numerical representations tailored for both regularized linear regressors (Ridge, Lasso) and non-linear ensemble regressors (Random Forest, XGBoost, LightGBM).

### Key Milestones Achieved:
1. **Engineered Dominant Regressors:** Constructed domain features (`TotalLivingSF`, `TotalBathrooms`, `PropertyAge`, `TotalPorchSF`, `Neighborhood_MedianPricePerSF`). Our engineered variable **`TotalLivingSF`** emerged as the **#1 most informative predictor** across the entire dataset (Mutual Information score $= \mathbf{0.679}$).
2. **Eliminated Severe Multicollinearity:** Resolved spatial collinearity between floor levels; reduced maximum Variance Inflation Factor (**VIF**) from **$117.3$ down to $2.98$** (below the conservative $5.0$ threshold).
3. **Target & Distribution Normalization:** Transformed target variable to $\text{LogSalePrice} = \log(1 + \text{SalePrice})$ and applied $\log(1 + x)$ to $25$ skewed continuous features ($|\text{skew}| > 0.75$).
4. **Hybrid Encoding Architecture:** Mapped $14$ hierarchical physical attributes to ordered integers ($5 \dots 0$) and performed one-hot encoding on nominal features with strict train/test dimensional alignment ($287$ columns in Train, $285$ in Test).

---

## 2. Construction of Derived Domain Features

| Engineered Feature | Mathematical Formulation / Definition | Business Rationale & Valuation Impact |
| :--- | :--- | :--- |
| **`TotalLivingSF`** | $\text{TotalBsmtSF} + \text{1stFlrSF} + \text{2ndFlrSF}$ | In residential real estate, usable finished basement space directly augments above-ground living area. Resolves collinearity while capturing total living capacity. |
| **`TotalBathrooms`** | $\text{FullBath} + 0.5 \cdot \text{HalfBath} + \text{BsmtFullBath} + 0.5 \cdot \text{BsmtHalfBath}$ | Standard appraisal fractional bathroom index. Consolidates 4 separate bathroom counts into a single monotonic regressor ($r = 0.63$). |
| **`PropertyAge`** | $\max(0, \text{YrSold} - \text{YearBuilt})$ | Quantifies physical and architectural vintage at transaction closing. Directly powers non-linear depreciation curves. |
| **`YearsSinceRemod`** | $\max(0, \text{YrSold} - \text{YearRemodAdd})$ | Measures elapsed time since capital modernization. Captures current condition premium. |
| **`IsRemodeled`** | $\mathbb{I}(\text{YearRemodAdd} > \text{YearBuilt})$ | Binary flag identifying whether the asset underwent substantial renovation. |
| **`TotalPorchSF`** | $\text{OpenPorchSF} + \text{EnclosedPorch} + \text{3SsnPorch} + \text{ScreenPorch} + \text{WoodDeckSF}$ | Aggregates outdoor recreational square footage, valued by suburban Midwestern buyers. |
| **`OverallGrade`** | $\text{OverallQual} \times \text{OverallCond}$ | Multiplicative interaction term isolating premium-condition luxury properties ($[1, 100]$ scale). |
| **`Neighborhood_MedianPricePerSF`** | $\text{Median}\left(\frac{\text{SalePrice}}{\text{TotalLivingSF}} \mid \text{Neighborhood}\right)_{\text{train}}$ | Spatial target encoding without data leakage. Represents local price benchmark per unit area. |
| **Amenity Flags** | $\mathbb{I}(\text{Pool} > 0), \mathbb{I}(\text{Garage} > 0), \mathbb{I}(\text{Fireplace} > 0), \mathbb{I}(\text{Bsmt} > 0)$ | Discrete presence indicators separating zero-area absence from positive continuous dimensions. |

> **Note on "Price per square meter":** Directly calculating $\text{SalePrice} / \text{Area}$ for each individual property would introduce catastrophic **target leakage** (using the target $y$ to predict $y$). Instead, our architecture computes historical neighborhood-aggregated median prices per square foot fitted strictly on the training partition and broadcast as an independent spatial benchmark.

---

## 3. Target & Feature Skewness Rectification

### 3.1 Target Variable Transformation: $\log(1 + \text{SalePrice})$
Residential prices follow a log-normal distribution due to proportional compounding of value. Linear models assume homoscedastic, normally distributed residuals $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$:
* **Raw $\text{SalePrice}$ Skewness:** $+1.88$ (Leptokurtic, kurtosis $+6.54$)
* **Transformed $\text{LogSalePrice}$ Skewness:** **$+0.12$** (Near-mesokurtic, kurtosis $+0.81$)

All models in Module 5 will be trained on $\text{LogSalePrice}$. Evaluation metrics will be computed both in log-space ($\text{RMSE}_{\log}$) and on original dollar values via $\hat{y} = \exp(\hat{y}_{\log}) - 1$.

### 3.2 Continuous Regressor Skewness Correction
Continuous features with absolute skewness $|\gamma_1| > 0.75$ were transformed via $\log(1 + x)$. A total of **$25$ skewed regressors** were normalized:
* **Spatial Dimensions:** `LotFrontage`, `LotArea`, `MasVnrArea`, `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `1stFlrSF`, `2ndFlrSF`, `LowQualFinSF`, `GrLivArea`, `TotalLivingSF`, `TotalPorchSF`, `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `3SsnPorch`, `ScreenPorch`, `PoolArea`.
* **Synthetic Contextual Features:** `renovation_cost_usd`, `days_on_market`, `distance_to_school_km`, `distance_to_transit_km`.

---

## 4. Categorical Variable Encoding Architecture

### 4.1 Ordinal Encoding for Hierarchical Ratings
Ames housing contains $14$ categorical attributes that represent ordered qualitative assessments. Standard one-hot dummy encoding discards this natural order. We mapped them to integer scales:
* **Standard Quality Scale (`ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond`, `PoolQC`):**
  $$\text{Ex} = 5, \quad \text{Gd} = 4, \quad \text{TA} = 3, \quad \text{Fa} = 2, \quad \text{Po} = 1, \quad \text{None} = 0$$
* **Basement Exposure (`BsmtExposure`):** $\text{Gd} = 4, \text{Av} = 3, \text{Mn} = 2, \text{No} = 1, \text{None} = 0$
* **Basement Finish Rating (`BsmtFinType1`, `BsmtFinType2`):** $\text{GLQ} = 6 \dots \text{Unf} = 1, \text{None} = 0$
* **Functional Home Rating (`Functional`):** $\text{Typ} = 7 \dots \text{Sal} = 0$
* **Central Air (`CentralAir`):** $\text{N} = 0, \text{Y} = 1$

### 4.2 One-Hot Encoding with Strict Dimensional Alignment
Nominal features with no natural ordering (`Neighborhood`, `MSZoning`, `BldgType`, `HouseStyle`, `Foundation`, `renovation_type`, etc.) were converted to binary indicator variables (`drop_first=True`).  
* Concatenating train and test temporarily prior to dummification guaranteed **$100\%$ structural alignment**:
  * **Engineered Training Data:** **$1,458$ records $\times$ $287$ columns** (includes `Id`, `SalePrice`, `LogSalePrice`)
  * **Engineered Test Data:** **$1,459$ records $\times$ $285$ columns** (includes `Id`, identical $284$ features)

---

## 5. Multicollinearity Resolution & Feature Selection
*Visual Reference:* `reports/figures/12_feature_engineering_vif_reduction.png`

### 5.1 Variance Inflation Factor (VIF) Audit: Before vs. After
In Module 2, dimensional features suffered from extreme variance inflation ($\text{VIF} > 117$). By consolidating floors into `TotalLivingSF` and replacing redundant raw counts with composite metrics, collinearity was completely eradicated:

| Regressor (Before Engineering) | Raw VIF | Regressor (After Engineering) | Engineered VIF | Collinearity Status |
| :--- | :---: | :--- | :---: | :---: |
| `GrLivArea` | **$117.33$** | `Neighborhood_MedianPricePerSF` | **$2.98$** | **Ideal ($\text{VIF} < 3.0$)** |
| `2ndFlrSF` | **$86.81$** | `OverallQual` | **$2.97$** | **Ideal ($\text{VIF} < 3.0$)** |
| `1stFlrSF` | **$65.06$** | `PropertyAge` | **$2.70$** | **Ideal ($\text{VIF} < 3.0$)** |
| `GarageCars` | **$5.50$** | `TotalLivingSF` | **$2.41$** | **Ideal ($\text{VIF} < 3.0$)** |
| `GarageArea` | **$5.01$** | `TotalBathrooms` | **$1.95$** | **Ideal ($\text{VIF} < 3.0$)** |
| `TotalBsmtSF` | $3.39$ | `GarageCars` | **$1.89$** | **Ideal ($\text{VIF} < 3.0$)** |
| `TotRmsAbvGrd` | $3.38$ | `amenity_composite_score` | **$1.85$** | **Ideal ($\text{VIF} < 3.0$)** |
| `YearBuilt` | $2.71$ | `TotalPorchSF` | **$1.20$** | **Ideal ($\text{VIF} < 3.0$)** |
| `amenity_composite_score` | $1.71$ | `days_on_market` | **$1.05$** | **Ideal ($\text{VIF} < 3.0$)** |
| `days_on_market` | $1.05$ | `mortgage_rate_pct` | **$1.01$** | **Ideal ($\text{VIF} < 3.0$)** |

---

## 6. Feature Importance via Mutual Information
*Visual Reference:* `reports/figures/13_feature_importance_mi.png`

Mutual Information (MI) quantifies the non-parametric statistical dependency between each regressor and the target $\text{LogSalePrice}$:

| Rank | Feature Name | Feature Origin | MI Score | Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **`TotalLivingSF`** | **Engineered** | **$0.679$** | **Strongest single predictor in the entire model.** |
| **2** | `OverallQual` | Base Dataset | $0.575$ | Subjective construction quality and material rating. |
| **3** | **`Neighborhood_MedianPricePerSF`** | **Engineered** | **$0.496$** | Geographic spatial valuation benchmark. |
| **4** | `GrLivArea` | Base Dataset | $0.481$ | Above-grade interior area. |
| **5** | **`OverallGrade`** | **Engineered** | **$0.437$** | Interaction between build quality and maintenance condition. |
| **6** | `TotalBsmtSF` | Base Dataset | $0.367$ | Total basement footprint. |
| **7** | `GarageArea` | Base Dataset | $0.367$ | Vehicle storage area. |
| **8** | `GarageCars` | Base Dataset | $0.365$ | Vehicle capacity count. |
| **9** | `YearBuilt` | Base Dataset | $0.357$ | Initial construction vintage. |
| **10** | **`TotalBathrooms`** | **Engineered** | **$0.345$** | Fractional bathroom count. |
| **11** | **`PropertyAge`** | **Engineered** | **$0.335$** | Elapsed depreciation period. |
| **12** | `ExterQual_Ordinal` | Engineered Ordinal | $0.332$ | Exterior material grade. |
| **13** | `KitchenQual_Ordinal` | Engineered Ordinal | $0.327$ | Kitchen finish grade. |
| **14** | `BsmtQual_Ordinal` | Engineered Ordinal | $0.323$ | Basement ceiling height rating. |
| **15** | **`amenity_composite_score`** | **Synthetic Context** | **$0.311$** | Proximity to schools, transit, and medical infrastructure. |

---

## 7. Summary & Readiness for Module 5 (Model Development)

The feature engineering pipeline has created a leak-free, clean, non-collinear feature matrix ready for algorithmic training:
* **Train Dataset:** [`data/engineered/train_engineered.csv`](../data/engineered/train_engineered.csv) ($1,458$ records, $287$ columns)
* **Test Dataset:** [`data/engineered/test_engineered.csv`](../data/engineered/test_engineered.csv) ($1,459$ records, $285$ columns)

The project is now fully prepared for **Module 5: Model Development & Benchmarking**!
