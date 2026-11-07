# Business Understanding & Valuation KPIs
**Project:** House Price Prediction for a Real Estate Listing Platform  
**Course:** Business Analytics - Module 1 Deliverable  
**Domain:** Real Estate Valuation & Algorithmic Pricing (Automated Valuation Models — AVM)

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Core Business Challenge
Online real estate listing platforms (such as Zillow, Redfin, or ImmoScout24) serve as the primary discovery and valuation gateway for residential buyers and sellers. When a homeowner decides to list a property, establishing an accurate, defensible estimate of the property's fair market value is critical:
* An **overly high asking price** discourages serious buyers, increases marketing duration, and often forces consecutive price cuts that stigmatize the listing.
* An **undervalued asking price** leaves substantial capital on the table for the seller and diminishes the platform's transaction fee potential.

### 1.2 The Core Business Question
> *„Given a property’s physical characteristics, condition, and location, what is its estimated market price, and how confident is that estimate?“*

To answer this question, our team acts as the internal **Valuation Analytics Function** of the platform, building an end-to-end regression engine that predicts property sale prices alongside calibrated prediction confidence intervals.

---

## 2. Stakeholder Impact & Error Cost Asymmetry


### 2.1 The Business Cost of Over-Valuation ($\hat{y} > y$)
1. **Liquidity Friction & High Days-on-Market (DOM):** The property fails to attract offers within the peak interest window (the first 14–21 days of listing).
2. **Stigmatization & Forced Price Cuts:** Properties that sit on the market accumulate high DOM, signaling potential unobservable defects to buyers. Sellers eventually capitulate to sharp price reductions, frequently below original fair market value.
3. **Erosion of Platform Trust:** Sellers who followed the platform's recommendation blame the portal for poor market performance.

### 2.2 The Business Cost of Under-Valuation ($\hat{y} < y$)
1. **Capital Loss ("Money Left on the Table"):** The homeowner sells below intrinsic value, resulting in immediate wealth destruction.
2. **Commission Fee Attrition:** The platform and partnered brokerage firms earn fees proportional to transaction value ($\text{Fee} = \text{Price} \times \text{Commission Rate}$). An undervalued listing directly suppresses revenue.
3. **Immediate Seller Churn:** Sophisticated sellers consult multiple portals. A platform that systematically undervalues assets loses exclusive listing contracts to competitors.

---

## 3. Quantitative Key Performance Indicators (KPIs) & Targets

To hold the analytics engine accountable to commercial standards, the modeling framework establishes strict error tolerance targets:

### 3.1 Primary Business Metrics (Relative Error)
* **MAPE (Mean Absolute Percentage Error):**
  $$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100$$
  * **Target Threshold:** $\text{MAPE} \le \mathbf{8.5\%}$ (Mirroring commercial on-market AVM standards).
* **MdAPE (Median Absolute Percentage Error):**
  * Robust against extreme luxury or distressed property outliers.
  * **Target Threshold:** $\text{MdAPE} \le \mathbf{6.0\%}$.

### 3.2 Accuracy Bands / Prediction Error Tolerance
Real estate platforms gauge performance by the proportion of properties predicted within defined error boundaries:
| Error Band | Definition | Target Percentage | Business Rationale |
| :--- | :--- | :--- | :--- |
| **P-5 (Within $\pm 5\%$)** | $|\hat{y} - y| / y \le 0.05$ | $\ge \mathbf{60.0\%}$ | Highly accurate pricing; immediate market acceptance. |
| **P-10 (Within $\pm 10\%$)** | $|\hat{y} - y| / y \le 0.10$ | $\ge \mathbf{85.0\%}$ | Standard commercial valuation tolerance. |
| **P-20 (Within $\pm 20\%$)** | $|\hat{y} - y| / y \le 0.20$ | $\ge \mathbf{96.0\%}$ | Avoids catastrophic mispricings that trigger customer churn. |

### 3.3 Statistical Regression Benchmarks
* **Coefficient of Determination ($R^2$):** $\ge \mathbf{0.88}$ on holdout cross-validation.
* **Root Mean Squared Error (RMSE):** $\le \mathbf{\$24,000}$ (Penalizes large errors heavily).
* **Mean Absolute Error (MAE):** $\le \mathbf{\$15,000}$.

### 3.4 Uncertainty & Confidence Quantification
In accordance with the prompt (*"and how confident is that estimate?"*), the deployed application cannot return an isolated point prediction $\hat{y}$. It must output a **calibrated 90% Prediction Interval**:
$$[\hat{y}_{\text{lower}}, \hat{y}_{\text{upper}}]$$
Where properties in well-covered, homogeneous neighborhoods (`CollgCr`, `NAmes`) exhibit tight confidence bands ($\pm 4\text{--}6\%$), and unique or sparse luxury estates exhibit wider ranges ($\pm 10\text{--}15\%$) signaling to the user that a human appraisal is advisable.
