# Module 4 Summary: Feature Engineering, Transformation & Selection

## 1. Overview & Objectives
The primary objective of **Module 4** is to transform the cleaned Ames Housing datasets (`train_cleaned.csv` and `test_cleaned.csv` from Module 3) into a model-ready feature matrix without introducing **data leakage**. All statistical parameters (medians, skewness thresholds, scaling factors, and selection rankings) are fitted strictly on the training set and subsequently applied to the test set.

* **Input Datasets:**
  * **Training Set:** 1,458 rows $\times$ 94 columns (8,794 missing values, primarily representing structural absences such as `No Pool` or `No Garage`).
  * **Test Set:** 1,459 rows $\times$ 93 columns (8,922 missing values).

---

## 2. Domain-Driven Feature Construction
Using `fe.construct_derived_features()`, **16 domain-specific features** were engineered to capture physical property attributes, temporal depreciation, spatial valuation, and non-linear quality interactions:

### A. Area & Spatial Valuation Metrics
* **`TotalSF`:** Combined total living area (`TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`).
* **`GrLivArea_SqM` & `TotalSF_SqM`:** Metric conversions of above-grade and total square footage ($1\text{ sq ft} = 0.092903\text{ m}^2$).
* **`Neighborhood_Median_PricePerSqM`:** Out-of-fold/train-only median price per square meter grouped by `Neighborhood` (overall median: **$\$1,296.40/\text{m}^2$**), providing a strong spatial baseline without leaking target values.

### B. Property Age & Renovation Indicators
* **`PropertyAge`:** Age of the house at the time of sale (`YrSold - YearBuilt`; mean: **36.6 years**).
* **`YearsSinceRemodel`:** Time elapsed since the last remodeling (`YrSold - YearRemodAdd`; mean: **23.0 years**).
* **`GarageAge`:** Age of the garage (`YrSold - GarageYrBlt`; mean: **31.4 years**).
* **`IsNewBuild`:** Binary indicator flagging newly constructed homes (`YearBuilt == YrSold`; **11.0%** of training properties).
* **`IsRenovated`:** Binary indicator for properties modified after initial construction (`YearRemodAdd > YearBuilt`; **55.0%** of training properties).
* **`RemodelSpanYears`:** Duration between original construction and remodeling (`YearRemodAdd - YearBuilt`).

### C. Amenities, Ratios & Interaction Terms
* **`TotalBathrooms`:** Weighted sum of full and half bathrooms across basement and above-grade levels ($\text{Full} + 0.5 \times \text{Half}$; mean: **2.21 bathrooms**).
* **`TotalPorchSF`:** Aggregated area of open porch, enclosed porch, 3-season porch, and screen porch spaces.
* **`LivingToLotRatio`:** Ratio of above-grade living area to total lot area (`GrLivArea / LotArea`), measuring land-use density.
* **`AmenityProximityScore`:** Composite index reflecting proximity to desirable or undesirable external conditions (`Condition1`, `Condition2`, and neighborhood characteristics).
* **`Qual_x_TotalSF` & `Qual_x_GrLivArea`:** Multiplicative interaction terms combining `OverallQual` with total and above-grade living areas to capture compounding valuation effects.

---

## 3. Data Transformation & Encoding Pipeline
To satisfy linear and distance-based model assumptions while preserving ordinal hierarchies, the following preprocessing transformations are applied:

1. **Target & Skewness Normalization:**
   * Applied $\log(1 + x)$ (`log1p`) transformation to the target variable `SalePrice` to correct right-skewness and stabilize variance.
   * Applied `log1p` transformation to all numeric predictors exhibiting high skewness ($|\text{skew}| > 0.75$, calculated exclusively on the training set).
2. **Ordinal Encoding:**
   * Mapped **20 ordinal quality and condition variables** (e.g., `ExterQual`, `BsmtQual`, `KitchenQual`, `FireplaceQu`, `GarageQual`) to explicit integer scales (`0 = None/Missing` up to `5 = Excellent`) to preserve natural rank ordering.
3. **Nominal One-Hot Encoding:**
   * Applied One-Hot Encoding (`drop_first=True` / aligned columns) to remaining nominal categorical features, ensuring identical feature schemas across train and test splits.
4. **Outlier-Resistant Scaling:**
   * Standardized continuous numerical predictors using `RobustScaler` (centering on the median and scaling by the Interquartile Range, $\text{IQR}$), minimizing sensitivity to residual housing market outliers.

---

## 4. Multicollinearity Reduction & Hybrid Feature Selection
To prevent overfitting and eliminate redundant predictors in the expanded feature space, a three-stage reduction pipeline is executed:

1. **Pairwise Correlation Filtering:**
   * Removed highly collinear features exceeding a Pearson correlation threshold of $|r| > 0.90$ (retaining the feature with stronger target correlation).
2. **Iterative Variance Inflation Factor (VIF) Pruning:**
   * Iteratively dropped predictors with $\text{VIF} > 10.0$ to resolve multi-feature linear dependencies.
3. **Hybrid Importance Ranking (Top-60 Selection):**
   * Ranked surviving features using a composite score combining:
     * **Random Forest Feature Importance** (non-linear tree-based splits)
     * **Mutual Information (MI)** (general statistical dependency with `log1p(SalePrice)`)
     * **Absolute Target Correlation** (linear association strength)
   * Retained the **Top 60 features** for downstream predictive modeling in Module 5.

---

## 5. Key Takeaways & Next Steps
* **Leakage-Free Design:** Every aggregation (`Neighborhood_Median_PricePerSqM`), threshold, scaler, and selector was fit strictly on `X_train` and `y_train`.
* **High Predictive Signal:** Newly constructed interaction terms (`Qual_x_TotalSF`, `TotalSF`) and spatial price baselines (`Neighborhood_Median_PricePerSqM`) substantially strengthen linear and non-linear relationships with `SalePrice`.
* **Next Step (Module 5):** Train, cross-validate, and tune regularized linear models (Ridge, Lasso, ElasticNet) and gradient-boosted decision trees (XGBoost, LightGBM, CatBoost) on the selected Top-60 feature set.