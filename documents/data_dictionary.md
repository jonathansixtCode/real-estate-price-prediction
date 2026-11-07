# Data Dictionary: Synthetic Contextual Extensions
**Project:** House Price Prediction for a Real Estate Listing Platform  
**Course:** Business Analytics — Module 1 Deliverable  
**Base Dataset:** Ames Housing Dataset (Ames, Iowa, 2006–2010)

---

## 1. Overview
In accordance with the project requirements for **Module 1 (Business Understanding & Data Generation)**, the original Ames Housing dataset has been augmented with synthetic contextual variables using Python (`src/data_generation.py`).

These features reflect real-world property valuation drivers that are absent from raw structural property records:
1. **Neighborhood Amenity Proximity** (School, Hospital, CyRide Public Transit)
2. **Renovation & Remodeling History** (Type, Financial Investment, Contractor, Permit)
3. **Marketing Dynamics** (Days-on-Market / Listing Duration)
4. **Macroeconomic Indicators** (Mortgage Interest Rates, Housing Price Index, Unemployment Rate)

---

## 2. Synthetically Generated Features

| Column Name | Data Type | Unit / Scale | Valid Range | Generation Logic & Business Assumptions |
| :--- | :--- | :--- | :--- | :--- |
| `distance_to_school_km` | `float64` | Kilometers (km) | $[0.20, 5.00]$ | Euclidean/street distance to the nearest elementary or high school. Calibrated per Ames neighborhood baseline (e.g. `OldTown`: $\approx 0.6$ km; outer zones like `Timber`: $\approx 3.5$ km) with Gaussian intra-neighborhood variance ($\sigma = 0.12$). |
| `distance_to_hospital_km` | `float64` | Kilometers (km) | $[0.50, 8.00]$ | Distance to Mary Greeley Medical Center (central Ames). Baseline distances derived from neighborhood proximity to central Ames with local jitter ($\sigma = 0.18$). |
| `distance_to_transit_km` | `float64` | Kilometers (km) | $[0.08, 2.50]$ | Distance to the nearest CyRide bus or transit hub. High density in university-adjacent zones (`SWISU`, `CollgCr`) and downtown; lower density in peripheral developments. |
| `amenity_composite_score` | `float64` | Index score $[0, 100]$ | $[25.0, 98.0]$ | Inversely weighted composite accessibility score: $\text{Score} = 100 - (8 \cdot d_{\text{school}} + 4.5 \cdot d_{\text{hospital}} + 25 \cdot d_{\text{transit}})$. Higher values reflect premium walkable/transit-accessible locations. |
| `has_renovation` | `int64` | Binary flag $\{0, 1\}$ | $\{0, 1\}$ | Indicates whether the property underwent substantial renovation. Set to $1$ if `YearRemodAdd > YearBuilt` and $\le \text{YrSold}$, plus a $25\%$ probability of cosmetic updates for unremodeled homes older than 20 years. |
| `renovation_type` | `object` / `category` | String category | `None`, `Cosmetic`, `Kitchen_Bath`, `Major_Structural` | Category of renovation performed. Probabilistic distribution conditioned on property quality (`OverallQual`): luxury properties undergo structural/kitchen remodels; lower-tier properties undergo cosmetic updates. |
| `renovation_cost_usd` | `float64` | US Dollars (\$) | $\$0.00$ to $\approx \$120,000$ | Estimated financial expenditure on renovations. Follows a right-skewed lognormal distribution conditioned on `renovation_type` and scaled by gross living area (`GrLivArea`). $0.00$ for unrenovated homes. |
| `renovation_contractor` | `object` / `string` | Company Name | Valid company name or `'None'` | Name of the licensed renovation general contractor, generated deterministically using the Python `Faker` library (`en_US` locale) or regional construction fallback registry. |
| `renovation_permit_id` | `object` / `string` | Alphanumeric code | Format: `PRM-<Year>-<ID>` or `'None'` | Official municipal building permit identifier associated with the remodeling year. |
| `days_on_market` | `int64` | Days | $[3, 365]$ | Duration the listing remained on the market prior to final closing. Follows a lognormal distribution ($\text{median} \approx 42$ days). Adjusted downward for high quality (`OverallQual`) and peak spring/summer selling season (`MoSold` 4–7); adjusted upward for poor condition (`OverallCond`) and winter sales. |
| `mortgage_rate_pct` | `float64` | Percentage (\%) | $[4.10\%, 6.80\%]$ | Historical monthly average for the US 30-Year Fixed Mortgage Rate (Freddie Mac PMMS) matched to the property's transaction month (`YrSold`, `MoSold`), with lender spread variation ($\pm 0.04\%$). |
| `local_price_index` | `float64` | Index (Base 100 = Jan 2006) | $[90.0, 105.0]$ | Regional House Price Index (HPI) capturing the macroeconomic real estate cycle in Iowa from the 2006 peak through the 2008–2009 recession and 2010 stabilization. |
| `unemployment_rate_pct` | `float64` | Percentage (\%) | $[3.5\%, 6.7\%]$ | Historical Iowa state monthly unemployment rate reflecting macro economic stress during the transaction period. |

---

## 3. Data Integrity & Consistency Rules

1. **Temporal Precedence:**
   $$\text{YearBuilt} \le \text{YearRemodAdd} \le \text{YrSold}$$
   Permit IDs and contractor records reference valid remodeling years that do not exceed the sale year.
2. **Quality-Cost Correlation:**
   Properties with higher `OverallQual` and larger `GrLivArea` exhibit proportionally higher renovation expenditure distributions, avoiding unrealistically high investments on distressed structures.
3. **Reproducibility:**
   All synthetic features are generated using `np.random.default_rng(seed=42)` ensuring deterministic, bit-for-bit identical replication across all development environments.
