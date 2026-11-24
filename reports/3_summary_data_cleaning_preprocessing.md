# Module 3 Summary: Data Cleaning & Preprocessing

## 1. Overview & Dataset Dimensions

In Module 3, the augmented datasets (`train_augmented.csv` and `test_augmented.csv`) were cleaned and preprocessed via the `dc.run_module3_cleaning(...)` pipeline. All statistical thresholds (imputation values, winsorization caps) were derived strictly from the training data and subsequently applied to the test set to prevent data leakage.

| Metric                                    | Train (Before) | Train (After) | Test (Before) | Test (After) | Change / Notes                        |
| :---------------------------------------- | :------------- | :------------ | :------------ | :----------- | :------------------------------------ |
| **Row Count ($n$)**               | 1,460          | 1,458         | 1,459         | 1,459        | -2 outliers removed from Train        |
| **Column Count**                    | 88             | 88            | 87            | 87           | All features preserved                |
| **Missing Values (Cells)**          | 9,824          | 0             | 9,978         | 0            | 100% resolved                         |
| **Exact / ID / Feature Duplicates** | 0              | 0             | 0             | 0            | No duplicates found                   |
| **Standardized Categories (Cells)** | —             | 4,233         | —            | 4,215        | Spelling & labels unified             |
| **Corrected Logic Errors**          | —             | > 0           | —            | > 0          | Invalid years & inconsistencies fixed |
| **Winsorized Values**               | —             | 16            | —            | 5            | `LotArea` & `TotalBsmtSF` capped  |

---

## 2. Detailed Cleaning Steps

### 2.1 Duplicate Check

* **Exact duplicates, duplicate `Id`s, and feature duplicates:** **0** duplicates were found in either dataset. No rows had to be dropped due to redundancy.

### 2.2 Categorical Standardization

Inconsistent spellings, whitespace, and divergent abbreviations were unified across numerous categorical variables (totaling **4,233 cells** in Train and **4,215 cells** in Test):

* **Exterior Covering (`Exterior1st` & `Exterior2nd`):** Standardized inconsistent labels (e.g., `Wd Shng` $\rightarrow$ `WdShing`, `Brk Cmn` $\rightarrow$ `BrkComm`, `CmentBd` $\rightarrow$ `CemntBd`).
* **Zoning, Building Type & Style:** Standardized `MSZoning` (`C (all)` $\rightarrow$ `C`), `BldgType` (`Twnhs` $\rightarrow$ `TwnhsE` / `Duplex`, etc.), and `HouseStyle`.
* **Augmented Features:** Cleaned and unified category labels in `renovation_type`.

### 2.3 Deterministic Logic & Data Entry Errors

Impossible or contradictory feature combinations were corrected using deterministic rules:

* **Implausible Construction Years (`GarageYrBlt > 2010`):** A typo in the test set (`2207`) was corrected to `2007`.
* **Temporal Inconsistencies (`YearRemodAdd < YearBuilt`, `YrSold < YearBuilt`):** Two cases with contradictory year entries in the test set were resolved.
* **Masonry Veneer (`MasVnrType` & `MasVnrArea`):** Resolved inconsistencies between zero area (`0`) and a specified veneer type (and vice versa).
* **Incomplete Garage & Basement Records:** Isolated records where individual sub-attributes were populated despite the absence of a garage or basement (or vice versa) were aligned for consistency.

### 2.4 Missing Value Treatment (Imputation)

All **9,824** missing values in the training set and **9,978** missing values in the test set were completely resolved:

1. **Structural Absence (Domain-Based):**
   * Missing values indicating the physical absence of a property feature were explicitly encoded: `'None'` for categorical features (e.g., `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, `GarageType`, `BsmtQual`, `MasVnrType`, `renovation_type`), `'NoPermit'` for missing building permits, and `0` for numerical area/count features (`GarageYrBlt`, `GarageArea`, `GarageCars`, `MasVnrArea`, basement areas/bathrooms).
2. **Statistical Imputation (Derived from Train Set):**
   * **Grouped Median:** `LotFrontage` was imputed using the median of the respective `Neighborhood` calculated on the training set.
   * **Mode Imputation (Most Frequent Value in Train):** Isolated true missing values in categorical variables were replaced with the training mode:
     * `Electrical`: `'SBrkr'`
     * `MSZoning`: `'RL'`
     * `Functional`: `'Typ'`
     * `Utilities`: `'AllPub'`
     * `Exterior1st` & `Exterior2nd`: `'VinylSd'`
     * `KitchenQual`: `'TA'`
     * `SaleType`: `'WD'`

---

## 3. Outlier Treatment & Skewness Reduction

### 3.1 Removal of Extreme Outliers (Train)

* **Criterion:** Living area `GrLivArea > 4,000` sq ft paired with a low sale price (`SalePrice < 300,000`).
* **Affected Records:** 2 partial-sale outliers in `Edwards` (`Id 524` and `Id 1299`).
* **Impact:** The maximum `GrLivArea` in Train dropped from **5,642** to **4,476**, reducing skewness from **1.367** to **1.011** (-26.0%).

### 3.2 Winsorization (99.5th Percentile of Train Set)

To mitigate extreme area values without discarding additional observations, `LotArea` and `TotalBsmtSF` were capped at the 99.5th percentile of the training data:

| Feature                   | Cap Value (99.5th Percentile) | Capped Rows (Train) | Capped Rows (Test) | Train Skewness (Before $\rightarrow$ After) |
| :------------------------ | :---------------------------- | :------------------ | :----------------- | :-------------------------------------------- |
| **`LotArea`**     | 53,192.80                     | 8                   | 1–2\*             | 12.574 $\rightarrow$ 3.814                  |
| **`TotalBsmtSF`** | 2,374.33                      | 8                   | 4                  | Significantly reduced                         |

*\*Documentation note: In the notebook output, the summary comparison table and the detailed audit log report slightly different figures for `LotArea` (12.208 vs. 12.574 for initial skewness, and 2 vs. 1 capped row in the test set).*

---

## 4. Quality Assurance & Modeling Readiness

* **Zero Missing Values & Consistent Data Types:** Both datasets are free of `NaN` values and ready for encoding and scaling in Module 4.
* **Data Leakage Prevention:** All imputation statistics (medians, modes) and winsorization thresholds were computed strictly on the cleaned training set.
* **Improved Distributions:** Removing the 2 partial-sale outliers and winsorizing extreme lot and basement areas substantially reduced right-skewness in key numerical predictors, benefiting linear models and distance-based algorithms in particular.
