# Data Dictionary: Synthetic Contextual Features

## 1. Overview & Purpose

To enrich the base Kaggle dataset (*"House Prices – Advanced Regression Techniques"*, Ames, Iowa; $n_{\text{train}} = 1,460$, $n_{\text{test}} = 1,459$) with realistic market and location context, **13 synthetic features** were generated programmatically using Python (`src/data_generation.py`, utilizing `numpy` and `Faker` with a fixed random seed for reproducibility).

These variables capture four key real-estate valuation dimensions not explicitly present in the raw structural dataset:
1. **Neighborhood Amenity Proximity** (distances to key infrastructure and composite accessibility score)
2. **Renovation & Modernization History** (historical upgrades, timing, and capital expenditure)
3. **Market Liquidity & Listing Dynamics** (time on market)
4. **Macroeconomic & Local Market Conditions** (interest rates, price indices, and employment context)

---

## 2. Synthetic Feature Specifications

| Column Name | Data Type | Unit | Valid Range | Generation Logic & Business Assumption |
| :--- | :--- | :--- | :--- | :--- |
| `distance_to_school_km` | `float64` | Kilometers ($\text{km}$) | $[0.2, 8.0]$ | Generated from a neighborhood-conditioned distribution (`Neighborhood` proximity tiers) with Gaussian noise, clipped to realistic city bounds. *Business assumption:* Family-oriented residential neighborhoods sit closer to schools, whereas rural/edge zones have longer commutes. |
| `distance_to_hospital_km` | `float64` | Kilometers ($\text{km}$) | $[0.5, 15.0]$ | Derived from spatial proximity tiers to Ames medical centers (e.g., Mary Greeley Medical Center) plus random variation. *Business assumption:* Central neighborhoods have faster emergency/healthcare access, supporting higher property desirability. |
| `distance_to_transit_km` | `float64` | Kilometers ($\text{km}$) | $[0.1, 6.0]$ | Simulated based on proximity to main transit corridors (CyRide bus network) and urban density of the neighborhood. *Business assumption:* Closer transit access increases convenience and rental/resale demand, especially for smaller or entry-level homes. |
| `amenity_composite_score` | `float64` | Score (index) | $[0.0, 100.0]$ | Weighted inverse-distance index combining `distance_to_school_km` (40%), `distance_to_transit_km` (35%), and `distance_to_hospital_km` (25%), normalized to a 0–100 scale. *Business assumption:* Buyers value overall locational convenience holistically rather than single amenities in isolation. |
| `has_renovation` | `int64` | Binary flag ($0 / 1$) | $\{0, 1\}$ | Bernoulli draw whose probability increases with property age (`YrSold - YearBuilt`) and is modulated by `OverallCond` and `YearRemodAdd > YearBuilt`. *Business assumption:* Older homes are substantially more likely to have undergone documented renovations prior to sale. |
| `years_since_renovation` | `float64` (`NaN` if unrenovated) | Years | $[0, 40]$ or `NaN` | If `has_renovation == 1`, sampled uniformly/triangularly between $0$ and $\min(\text{property age}, 40)$ (aligned with `YrSold - YearRemodAdd` where available); set to `NaN` when `has_renovation == 0`. *Business assumption:* Renovation value decays over time; recent upgrades command a stronger price premium. |
| `renovation_type` | `string` (`object`) | Category | `None`, `Cosmetic`, `Kitchen_Bath`, `Structural`, `Full_Remodel` | Assigned as `None` if `has_renovation == 0`; otherwise sampled conditionally on property quality (`OverallQual`) and living area (`GrLivArea`). *Business assumption:* High-end homes are more likely to undergo full or kitchen/bath remodels, whereas mid/lower-tier homes more frequently receive cosmetic updates. |
| `renovation_cost_usd` | `float64` | US Dollars ($\text{USD}$) | $[0.0, 150,000.0]$ | Set to `0.0` if `has_renovation == 0`; otherwise scaled by `GrLivArea` and `renovation_type` cost multipliers (e.g., `Cosmetic` < `Kitchen_Bath` < `Structural` < `Full_Remodel`) with log-normal noise. *Business assumption:* Larger homes and deeper structural interventions require proportionally higher capital investment. |
| `days_on_market` | `int64` | Days | $[1, 365]$ | Modeled via a negative-binomial / gamma-Poisson process influenced by `SaleCondition` (e.g., abnormal/foreclosure vs. normal), season (`MoSold`), and relative pricing/quality tier. *Business assumption:* Overpriced, lower-condition, or winter listings remain on the market longer than well-priced spring/summer listings. |
| `mortgage_rate_pct` | `float64` | Percent ($\%$) | $[3.50, 7.50]$ | Mapped from the transaction year and month (`YrSold`, `MoSold`, spanning 2006–2010) to historical 30-year fixed mortgage rate regimes with minor local lender spread noise. *Business assumption:* Higher prevailing mortgage rates reduce buyer purchasing power and exert downward pressure on transaction prices. |
| `local_price_index` | `float64` | Index ($2006\text{ Q1} = 100$) | $[85.0, 115.0]$ | Time-series index anchored at $100.0$ in early 2006, reflecting pre-2008 housing expansion, the 2008–2009 macroeconomic downturn, and subsequent stabilization, with subtle neighborhood-tier adjustments. *Business assumption:* Macro housing cycles shift baseline market valuations across all property types over time. |
| `unemployment_rate_pct` | `float64` | Percent ($\%$) | $[2.5, 8.5]$ | Tied to `YrSold` and `MoSold` reflecting Ames/Story County historical labor market trends (low ~3–4% pre-2008, rising during the 2008–2010 recession) plus minor monthly noise. *Business assumption:* Rising local unemployment dampens housing demand and increases time-on-market. |
| `listing_agent_id` | `string` (`object`) | Identifier | `AGT-001` – `AGT-050` | Generated using `Faker` / deterministic ID pool representing 50 active local brokerage agents assigned across listings. *Business assumption:* Captures realistic brokerage metadata for operational tracking (excluded or target-encoded during modeling to prevent high-cardinality overfitting). |

---

## 3. Data Integrity & Leakage Safeguards

* **Train/Test Consistency:** All synthetic generation parameters, neighborhood proximity mappings, and macroeconomic lookup tables are applied consistently across both `train_augmented.csv` and `test_augmented.csv` without using target (`SalePrice`) leakage.
* **Missing Value Semantics:** Missing values in `years_since_renovation` (`NaN`) and `renovation_type` (`None`) are structural (indicating `has_renovation == 0`) and are explicitly handled during Module 3 (`src/data_cleaning.py`).
* **Reproducibility:** Controlled via a fixed NumPy/Faker random seed in `src/data_generation.py`, ensuring identical synthetic datasets across pipeline runs.