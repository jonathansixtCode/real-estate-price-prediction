# Module 2: Exploratory Data Analysis (EDA) – Key Findings & Results

## 1. Dataset Overview & Structure
* **Dimensions:** The dataset comprises **1,460 residential property observations** across **81 features** (38 numerical, 43 categorical), with `SalePrice` serving as the continuous target variable.
* **Data Completeness:** While most core structural features are complete, several variables exhibit high proportions of missing values. In most cases (e.g., `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`), `NaN` values represent the *absence* of a feature rather than missing data, requiring domain-specific imputation (`"None"`) rather than deletion.

---

## 2. Target Variable Analysis (`SalePrice`)
* **Distribution Shape:** `SalePrice` exhibits a strong **positive (right) skewness** ($\text{Skewness} \approx 1.88$) and high **leptokurtic peakedness** ($\text{Kurtosis} \approx 6.54$), with a median price of $\$163,000$ and a mean of $\$180,921$.
* **Modeling Implication:** Because linear and gradient-boosted models perform better when target errors are normally distributed and homoscedastic, applying a **log-transformation** ($\log(1 + \text{SalePrice})$) is necessary to normalize the target distribution and stabilize variance.

---

## 3. Feature Distributions & Skewness
* **Continuous Area & Lot Features:** Features such as `LotArea` ($\text{Skewness} \approx 12.21$) and `GrLivArea` ($\text{Skewness} \approx 1.37$) are heavily right-skewed due to a small number of expansive properties.
* **Discrete & Sparse Features:** Features like `PoolArea`, `LowQualFinSF`, and `3SsnPorch` are zero-inflated (mostly $0$), acting more like binary indicators of feature presence rather than continuous predictors.

---

## 4. Outlier & Anomaly Detection (`GrLivArea` vs. `SalePrice`)
A bivariate analysis of above-grade living area (`GrLivArea`) against `SalePrice` revealed **four extreme properties** exceeding $4,000\text{ sq ft}$:

| Property ID | Neighborhood | `GrLivArea` (sq ft) | `SalePrice` | Sale Condition | Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1299** | Edwards | $5,642$ | $\$160,000$ | Partial | **Extreme Anomaly (Remove)** – Massive area at a disproportionately low price; distorts linear trends. |
| **524** | Edwards | $4,676$ | $\$184,750$ | Partial | **Extreme Anomaly (Remove)** – Same structural pattern as ID 1299; high leverage outlier. |
| **1183** | NoRidge | $4,476$ | $\$745,000$ | Abnorml | **Valid Luxury Property (Keep)** – High price aligns with the affluent `NoRidge` neighborhood. |
| **692** | NoRidge | $4,316$ | $\$755,000$ | Normal | **Valid Luxury Property (Keep)** – Follows the expected price-to-size trajectory. |

* **Actionable Decision:** Observations **1299** and **524** should be removed prior to model training to prevent severe distortion of regression slopes, while **1183** and **692** must be retained to preserve valid upper-end market representation.

---

## 5. Key Takeaways for Feature Engineering & Modeling
1. **Target & Feature Transformations:** Apply $\log(1 + x)$ transformations to `SalePrice` and heavily skewed numerical predictors (`LotArea`, `GrLivArea`, `TotalBsmtSF`) to satisfy normality assumptions.
2. **Domain-Informed Imputation:** Distinguish between true missingness (e.g., `LotFrontage`, which should be imputed via neighborhood median) and structural absence (`NaN` $\rightarrow$ `"None"` for garages, basements, and amenities).
3. **Outlier Filtering:** Drop the two anomalous `Edwards` partial-sale records (`GrLivArea > 4000` and `SalePrice < 300000`) during preprocessing.
4. **Multicollinearity Management:** Monitor highly correlated feature pairs (e.g., `GarageCars` vs. `GarageArea`, `TotalBsmtSF` vs. `1stFlrSF`, and `OverallQual` vs. neighborhood tiers) using Variance Inflation Factor (VIF) analysis in subsequent steps.