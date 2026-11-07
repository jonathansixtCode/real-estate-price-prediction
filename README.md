# Real Estate House Price Prediction Platform
*An End-to-End Analytics Project: From Raw Data to a Monitored, Deployed Model*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Module 1](https://img.shields.io/badge/Module%201-Completed-brightgreen.svg)]()
[![Module 2](https://img.shields.io/badge/Module%202-Completed-brightgreen.svg)]()
[![Module 3](https://img.shields.io/badge/Module%203-Completed-brightgreen.svg)]()
[![Module 4](https://img.shields.io/badge/Module%204-Completed-brightgreen.svg)]()
[![Checkpoint 1-3](https://img.shields.io/badge/Checkpoint%201--3-PASSED-success.svg)]()

---

## Project Overview
This repository contains the valuation-analytics engine for an automated real estate pricing platform (AVM). The goal is to estimate property market prices along with confidence intervals to help sellers set competitive listing prices and prevent capital loss or market stagnation.

---

## Modules Overview & Current Progress

| Module | Focus Area | Status | Deliverables & Code | Key Milestone Output |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Business Understanding & Data Generation** | **Done** | • [`documents/business_understanding.md`](documents/business_understanding.md)<br>• [`documents/data_dictionary.md`](documents/data_dictionary.md)<br>• [`src/data_generation.py`](src/data_generation.py)<br>• [`src/verify_data.py`](src/verify_data.py) | • AVM business problem defined ($\text{MAPE} \le 8.5\%$, 90% confidence range)<br>• 13 synthetic features generated (Amenities, Renovation, DOM, Macro)<br>• Bit-for-bit deterministic reproducibility (`seed=42`) |
| **2** | **Exploratory Data Analysis (EDA)** | **Done** | • [`notebooks/01_exploratory_data_analysis.ipynb`](notebooks/01_exploratory_data_analysis.ipynb)<br>• [`documents/eda_report.md`](documents/eda_report.md)<br>• [`src/eda_analysis.py`](src/eda_analysis.py)<br>• [`reports/figures/`](reports/figures/) (10 plots) | • Target skewness analyzed ($+1.88 \rightarrow +0.12$ under log1p)<br>• Outliers identified (`Id = 524, 1299`)<br>• VIF multicollinearity audit completed |
| **3** | **Data Cleaning & Quality Assurance** | **Done** | • [`notebooks/02_data_cleaning.ipynb`](notebooks/02_data_cleaning.ipynb)<br>• [`documents/data_cleaning_report.md`](documents/data_cleaning_report.md)<br>• [`src/data_cleaning.py`](src/data_cleaning.py)<br>• [`reports/figures/11_cleaning_before_after.png`](reports/figures/11_cleaning_before_after.png) | • Missing entries reduced from 9,809 to **0 missing values**<br>• Zero data leakage: estimators fitted strictly on Train<br>• Outliers removed; typos standardized (`C (all)`, `GarageYrBlt`) |
| **4** | **Feature Engineering & Preprocessing** | **Done** | • [`notebooks/03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb)<br>• [`documents/feature_engineering_report.md`](documents/feature_engineering_report.md)<br>• [`src/feature_engineering.py`](src/feature_engineering.py)<br>• [`reports/figures/12_feature_engineering_vif_reduction.png`](reports/figures/12_feature_engineering_vif_reduction.png)<br>• [`reports/figures/13_feature_importance_mi.png`](reports/figures/13_feature_importance_mi.png) | • `TotalLivingSF` engineered (#1 predictor, $\text{MI} = 0.679$)<br>• Multicollinearity eradicated (Max $\text{VIF} < 3.0$)<br>• Log1p on target & 25 skewed features<br>• Ordinal & One-hot encoding ($285$ aligned features) |
| **5** | **Model Development & Benchmarking** | *Next* | Linear/Ridge/Lasso, Random Forest, XGBoost/LightGBM | Cross-validation, RMSE, MAE, MAPE, $R^2$ |

---

## Project Structure
```text
real-estate-price-prediction/
├── data/
│   ├── train.csv                      # Raw Ames training set
│   ├── test.csv                       # Raw Ames test set
│   ├── data_description.txt           # Ames dataset feature documentation
│   ├── processed/                     # Module 1 Output: Augmented data
│   │   ├── train_augmented.csv        # 1,460 records x 94 columns (with synthetic features)
│   │   └── test_augmented.csv         # 1,459 records x 93 columns (with synthetic features)
│   ├── cleaned/                       # Module 3 Output: Cleaned data
│   │   ├── train_cleaned.csv          # 1,458 records x 94 columns (0 missing, outliers removed)
│   │   └── test_cleaned.csv           # 1,459 records x 93 columns (0 missing values)
│   └── engineered/                    # Module 4 Output: Modeling-ready features
│       ├── train_engineered.csv       # 1,458 records x 287 columns (with LogSalePrice)
│       └── test_engineered.csv        # 1,459 records x 285 columns (aligned feature matrix)
├── documents/
│   ├── RealEstate_House_Price_Prediction_Project_Announcement.pdf
│   ├── business_understanding.md      # Module 1 Task 1: Problem & KPI definition
│   ├── data_dictionary.md            # Module 1 Task 3: Complete data dictionary
│   ├── eda_report.md                 # Module 2 Tasks 1-3: Comprehensive EDA Report
│   ├── data_cleaning_report.md       # Module 3 Tasks 1-3: Data Cleaning Audit Report
│   └── feature_engineering_report.md # Module 4 Tasks 1-3: Feature Engineering Report
├── reports/
│   ├── eda_summary_metrics.json       # Quantitative metrics from EDA
│   ├── cleaning_before_after_metrics.json # Data cleaning audit metrics
│   ├── feature_engineering_metrics.json   # VIF reduction & MI feature importance metrics
│   └── figures/                       # 13 publication-quality visual plots (.png)
├── src/
│   ├── data_generation.py             # Module 1: Synthetic data pipeline
│   ├── verify_data.py                 # Module 1: Sanity validation suite
│   ├── eda_analysis.py                # Module 2: EDA visualization pipeline
│   ├── build_eda_notebook.py          # Module 2: Notebook generator
│   ├── data_cleaning.py               # Module 3: End-to-end cleaning pipeline
│   ├── build_cleaning_notebook.py     # Module 3: Cleaning notebook generator
│   ├── feature_engineering.py         # Module 4: Feature engineering & encoding pipeline
│   └── build_fe_notebook.py           # Module 4: Feature engineering notebook generator
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb # Module 2: Interactive EDA
│   ├── 02_data_cleaning.ipynb             # Module 3: Interactive Data Cleaning
│   └── 03_feature_engineering.ipynb       # Module 4: Interactive Feature Engineering
├── models/                            # Upcoming Module 5: Trained model artifacts
├── app/                               # Upcoming Module 6: Deployment (Streamlit / API)
└── monitoring/                        # Upcoming Module 7: Drift monitoring (Evidently)
```

---

## Quickstart & Replication

```bash
# 1. Generate synthetic features (Module 1)
python src/data_generation.py --data-dir data --output-dir data/processed --seed 42

# 2. Run EDA analysis (Module 2)
python src/eda_analysis.py

# 3. Clean raw datasets (Module 3)
python src/data_cleaning.py

# 4. Engineer features & encode matrices (Module 4)
python src/feature_engineering.py
```
