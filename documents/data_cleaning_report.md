# Data Cleaning & Preprocessing Report
**Project:** House Price Prediction for a Real Estate Listing Platform  
**Course:** Business Analytics — Module 3 Deliverable & Module 1–3 Milestone Checkpoint  
**Analyzed Datasets:** `data/processed/train_augmented.csv` & `data/processed/test_augmented.csv`  
**Output Cleaned Datasets:** [`data/cleaned/train_cleaned.csv`](../data/cleaned/train_cleaned.csv) & [`data/cleaned/test_cleaned.csv`](../data/cleaned/test_cleaned.csv)  
**Notebook Reference:** [`notebooks/02_data_cleaning.ipynb`](../notebooks/02_data_cleaning.ipynb)  
**Figure Reference:** [`reports/figures/11_cleaning_before_after.png`](../reports/figures/11_cleaning_before_after.png)

---

## 1. Executive Summary
This report documents the methodological execution of **Module 3 (Data Cleaning)**. Data cleaning bridges raw data exploration (Module 2) and advanced feature engineering (Module 4) by rectifying data inconsistencies, eliminating distorting leverage outliers, and resolving missingness across 94 features without introducing data leakage.

### Key Milestones Achieved:
1. **Zero Missingness:** Successfully reduced $9,809$ raw missing entries in training data and $9,918$ in test data to **$0$ missing entries**, preserving structural domain semantics.
2. **Elimination of Severe Leverage Outliers:** Excised two severe partial contract sales (`Id = 524` and `Id = 1299`) from the training set ($1,460 \rightarrow 1,458$ records), eliminating artificial slope depression on living area.
3. **Data Integrity & Typo Rectification:** Corrected an impossible future date entry (`GarageYrBlt = 2207`), standardized zoning labels (`MSZoning = 'C (all)'` $\rightarrow$ `'C'`), and audited record duplicates.
4. **Leakage-Free Architecture:** All statistical estimators (medians, modes) were fitted strictly on training data and broadcast to test data.

---

## 2. Before vs. After Quantitative Audit Table

| Dimension / Metric | Before Cleaning (Module 1/2 State) | After Cleaning (Module 3 Output) | Treatment Methodology |
| :--- | :---: | :---: | :--- |
| **Training Records ($n_{\text{train}}$)** | $1,460$ | **$1,458$** | Removed 2 severe partial sales outliers (`Id 524, 1299`). |
| **Test Records ($n_{\text{test}}$)** | $1,459$ | **$1,459$** | Strictly preserved all test instances for benchmark scoring. |
| **Total Features ($p$)** | $94$ (Train) / $93$ (Test) | **$94$ (Train) / $93$ (Test)** | Structural dimensionality maintained intact. |
| **Total Missing Values (Train)** | **$9,809$** | **$0$** | Domain-specific structural mapping + neighborhood medians. |
| **Total Missing Values (Test)** | **$9,918$** | **$0$** | Leakage-free broadcast of training estimators. |
| **Duplicate Observations** | $0$ exact duplicates | **$0$ exact duplicates** | Verified across all 93 feature columns. |
| **MSZoning Typo (`'C (all)'`)** | $25$ records ($10$ train, $15$ test) | **$0$ records** | Harmonized to official zoning code `'C'`. |
| **Garage Year Typo (`> 2026`)** | $1$ record (`Id 2593 = 2207`) | **$0$ records** | Corrected to property `YearBuilt` ($2007$). |

---

## 3. Detailed Data Cleaning Decisions

### 3.1 Duplicate Record Audit
A uniqueness audit across all $93$ non-ID regressors confirmed that no duplicate property listings exist in either the training or test partitions ($\text{duplicates} = 0$). Each record represents an independent transaction in Story County, Iowa.

### 3.2 Inconsistent Categorical Labels & Typo Correction
1. **Commercial Zoning Harmonization:** In Ames municipal records, Commercial Zoning is designated as `C`. In the raw data, $25$ instances ($10$ train, $15$ test) were recorded as `C (all)`. These were harmonized to `C` to prevent arbitrary one-hot encoding feature fragmentation.
2. **Impossible Temporal Value (`GarageYrBlt = 2207`):** Observation `Id 2593` in the test set had a garage construction year recorded as $2207$. Cross-referencing verified the primary structure was built in $2006$ and remodeled in $2007$. The garage year was corrected to $2007$ (`YearBuilt`).
3. **Categorical Whitespace Normalization:** All string attributes underwent `.str.strip()` to eliminate inadvertent leading or trailing whitespace.

### 3.3 Domain-Specific Missing Value Imputation Framework

In real estate, missing data is rarely Missing Completely at Random (MCAR). Missingness predominantly signals the **physical absence** of an amenity:

```
                            Missing Value Strategy
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
   STRUCTURAL ABSENCE                                    TRUE MISSINGNESS
 (Feature does not exist)                               (Unrecorded data)
            │                                                   │
  ┌─────────┴─────────┐                               ┌─────────┴─────────┐
  ▼                   ▼                               ▼                   ▼
Categoricals        Numericals                   LotFrontage          Categoricals
(Alley, Garage,    (Basement SF,              (Neighborhood-          (Electrical,
Pool, Fireplace)   Garage Cars)               stratified medians)     Zoning, Mode)
  │                   │                               │                   │
'None'               0.0                         Fitted on Train         Mode from Train
```

#### A. Structural Absence Categoricals $\rightarrow$ `'None'`
Variables where `NA` indicates that the house lacks the amenity were populated with `'None'`:
* `Alley` ($1,369$ train / $1,352$ test missing)
* Basement: `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2` ($37\text{--}38$ missing)
* `FireplaceQu` ($690$ train / $730$ test missing)
* Garage: `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond` ($81$ train / $78$ test missing)
* `PoolQC` ($1,453$ train / $1,456$ test missing)
* `Fence` ($1,179$ train / $1,169$ test missing)
* `MiscFeature` ($1,406$ train / $1,408$ test missing)

#### B. Structural Absence Numericals $\rightarrow$ `0.0`
Variables corresponding to unbuilt physical dimensions were populated with `0.0`:
* `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`
* `BsmtFullBath`, `BsmtHalfBath`
* `GarageCars`, `GarageArea`
* `MasVnrArea` (Masonry veneer area; set to $0.0$, accompanied by `MasVnrType = 'None'`)
* `GarageYrBlt` (Populated with $0.0$, maintaining a discrete flag for unbuilt garages)

#### C. `LotFrontage` (Neighborhood Stratified Median Imputation)
* **Problem:** $259$ training ($17.7\%$) and $227$ test records ($15.6\%$) lacked linear street exposure measurements. Imputing a global mean/median would distort lot geometries between dense urban lots (`MeadowV` median $= 21$ ft) and spacious executive lots (`NoRidge` median $= 91$ ft).
* **Solution:** We computed the median `LotFrontage` grouped by `Neighborhood` strictly on the training partition:
  $$\text{LotFrontage}_{i} = \text{Median}\left(\text{LotFrontage} \mid \text{Neighborhood}_i\right)_{\text{train}}$$
* **Leakage Prevention:** The training medians lookup dictionary was mapped directly onto the test set without re-computing test-set medians.

#### D. True Missing Categoricals $\rightarrow$ Mode Imputation
For isolated unrecorded entries, training modes were imputed:
* `Electrical`: $1$ missing in train $\rightarrow$ `'SBrkr'` (Standard Circuit Breakers, $91.4\%$ frequency)
* `MSZoning`: $4$ missing in test $\rightarrow$ `'RL'` (Residential Low Density)
* `Utilities`: $2$ missing in test $\rightarrow$ `'AllPub'`
* `Exterior1st` & `Exterior2nd`: $1$ missing in test $\rightarrow$ `'VinylSd'`
* `KitchenQual`: $1$ missing in test $\rightarrow$ `'TA'` (Typical/Average)
* `Functional`: $2$ missing in test $\rightarrow$ `'Typ'` (Typical)
* `SaleType`: $1$ missing in test $\rightarrow$ `'WD'` (Warranty Deed)

---

## 4. Outlier Diagnostics & Removal Decision
*Visual Audit:* `reports/figures/11_cleaning_before_after.png` (Panels B & C)

### 4.1 Identification of Severe Leverage Observations
During Module 2 EDA, two severe anomalies were highlighted in the above-grade living area:
1. **`Id = 524`:** `GrLivArea = 4,676 sq ft`, `OverallQual = 10`, `SalePrice = $184,750`
2. **`Id = 1299`:** `GrLivArea = 5,642 sq ft`, `OverallQual = 10`, `SalePrice = $160,000`

### 4.2 Economic & Statistical Justification for Dropping:
* **Partial Contract Sales:** Both properties carry `SaleCondition = 'Partial'`, indicating that the recorded transaction was an uncompleted builder's contract or estate liquidation rather than a fair market arm's-length sale.
* **Leverage & Influence:** In OLS and gradient-boosted trees, points with Cook's distance $D_i > 1$ exert disproportionate pull. These two observations flattened the linear slope of square footage by $\approx 14\%$, artificially penalizing predictions on standard luxury builds.
* **Decision:** Dropped strictly from the training dataset. Test instances are never discarded.

---

## 5. Verification of Cleaned Datasets

The final processed datasets were exported and verified:
* **Train Cleaned:** [`data/cleaned/train_cleaned.csv`](../data/cleaned/train_cleaned.csv) — **$1,458$ records, $94$ columns, $0$ missing values**.
* **Test Cleaned:** [`data/cleaned/test_cleaned.csv`](../data/cleaned/test_cleaned.csv) — **$1,459$ records, $93$ columns, $0$ missing values**.

---

## 6. Module 1–3 Checkpoint Sign-Off

With the completion of Module 3, our team has achieved **100% completion of the Module 1–3 Milestone Checkpoint**:
1. **Module 1 (Business Understanding & Synthetic Data):** Formulated AVM business question, quantified asymmetric error costs, generated 13 synthetic features with zero missingness, and produced the formal Data Dictionary.
2. **Module 2 (Exploratory Data Analysis):** Analyzed target skewness, geographic price dispersion, non-linear depreciation curves, macroeconomic trends, and identified multicollinear VIF clusters.
3. **Module 3 (Data Cleaning):** Completely resolved all missing values ($0$ nulls remaining), eliminated extreme leverage outliers, corrected data entry typos, and preserved complete data leakage separation.

The repository is now fully prepared for **Module 4: Feature Engineering**!
