"""
Module 1: Business Understanding & Synthetic Contextual Data Generation
Real Estate House Price Prediction Platform (Ames, Iowa)

Generates realistic supplementary contextual features for the Ames Housing dataset:
1. Neighborhood Amenity Proximity (distances to school, hospital, transit + composite score)
2. Renovation History (renovation flag, type, estimated cost, contractor via Faker, permit ID)
3. Marketing Dynamics (days on market based on property quality, condition, and seasonality)
4. Macroeconomic Indicators (30-year mortgage rate, local house price index, unemployment rate)

All random operations use a fixed seed (seed=42) for bit-for-bit reproducibility.
"""

import os
import sys
import math
import random
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Module1_DataGeneration")

NEIGHBORHOOD_BASELINES: Dict[str, Dict[str, float]] = {
    "OldTown": {"school": 0.60, "hospital": 0.85, "transit": 0.25},
    "BrkSide": {"school": 0.80, "hospital": 1.10, "transit": 0.30},
    "SWISU":   {"school": 1.20, "hospital": 3.20, "transit": 0.20},
    "Crawfor": {"school": 1.40, "hospital": 2.50, "transit": 0.40},
    "IDOTRR":  {"school": 0.90, "hospital": 1.50, "transit": 0.30},
    "Edwards": {"school": 1.10, "hospital": 3.00, "transit": 0.35},
    "CollgCr": {"school": 1.30, "hospital": 4.20, "transit": 0.40},
    "NAmes":   {"school": 0.70, "hospital": 2.20, "transit": 0.35},
    "NWAmes":  {"school": 0.95, "hospital": 2.80, "transit": 0.45},
    "Sawyer":  {"school": 1.15, "hospital": 3.60, "transit": 0.50},
    "SawyerW": {"school": 1.50, "hospital": 4.50, "transit": 0.55},
    "Gilbert": {"school": 1.80, "hospital": 4.80, "transit": 0.85},
    "NridgHt": {"school": 1.90, "hospital": 4.90, "transit": 0.90},
    "NoRidge": {"school": 2.10, "hospital": 4.60, "transit": 0.95},
    "Somerst": {"school": 1.60, "hospital": 3.90, "transit": 0.65},
    "StoneBr": {"school": 2.50, "hospital": 5.50, "transit": 1.20},
    "Timber":  {"school": 3.20, "hospital": 6.20, "transit": 1.50},
    "Mitchel": {"school": 2.40, "hospital": 5.10, "transit": 1.10},
    "ClearCr": {"school": 2.20, "hospital": 4.40, "transit": 1.00},
    "Veenker": {"school": 1.70, "hospital": 3.80, "transit": 0.80},
    "BrDale":  {"school": 1.10, "hospital": 2.60, "transit": 0.40},
    "NPkVill": {"school": 1.05, "hospital": 2.50, "transit": 0.38},
    "Blmngtn": {"school": 1.75, "hospital": 4.30, "transit": 0.75},
    "Blueste": {"school": 1.35, "hospital": 3.40, "transit": 0.45},
    "MeadowV": {"school": 2.60, "hospital": 5.40, "transit": 1.15},
}

MACRO_BY_YEAR: Dict[int, Dict[str, float]] = {
    2006: {"rate": 6.41, "hpi": 102.5, "unemp": 3.7},
    2007: {"rate": 6.34, "hpi": 103.8, "unemp": 3.8},
    2008: {"rate": 6.03, "hpi": 99.4,  "unemp": 4.2},
    2009: {"rate": 5.04, "hpi": 93.2,  "unemp": 6.1},
    2010: {"rate": 4.69, "hpi": 92.0,  "unemp": 6.0},
}

FALLBACK_CONTRACTORS = [
    "Heartland Home Builders LLC",
    "Ames Quality Renovations Inc.",
    "Story County Remodeling Co.",
    "Prairie Ridge Construction",
    "Midwest Custom Interiors",
]

SYNTHETIC_COLUMNS = [
    "distance_to_school_km",
    "distance_to_hospital_km",
    "distance_to_transit_km",
    "amenity_composite_score",
    "has_renovation",
    "renovation_type",
    "renovation_cost_usd",
    "renovation_contractor",
    "renovation_permit_id",
    "days_on_market",
    "mortgage_rate_pct",
    "local_price_index",
    "unemployment_rate_pct",
]


def generate_contextual_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Appends 13 deterministic synthetic contextual features to an Ames housing DataFrame."""
    rng = np.random.default_rng(seed)
    if FAKER_AVAILABLE:
        fake = Faker("en_US")
        Faker.seed(seed)
        contractor_pool = [fake.company() for _ in range(25)]
    else:
        contractor_pool = FALLBACK_CONTRACTORS

    records = []
    for idx, row in df.iterrows():
        nb = str(row.get("Neighborhood", "NAmes"))
        base = NEIGHBORHOOD_BASELINES.get(nb, {"school": 1.50, "hospital": 3.50, "transit": 0.60})

        # 1. Spatial distances & composite score
        d_school = float(np.clip(base["school"] + rng.normal(0.0, 0.12), 0.20, 5.00))
        d_hosp = float(np.clip(base["hospital"] + rng.normal(0.0, 0.18), 0.50, 8.00))
        d_trans = float(np.clip(base["transit"] + rng.normal(0.0, 0.08), 0.08, 2.50))

        raw_score = 100.0 - (8.0 * d_school + 4.5 * d_hosp + 25.0 * d_trans)
        amenity_score = float(np.clip(round(raw_score, 2), 25.0, 98.0))

        # 2. Renovation history
        yr_built = int(row.get("YearBuilt", 1980))
        yr_remod = int(row.get("YearRemodAdd", yr_built))
        yr_sold = int(row.get("YrSold", 2008))
        mo_sold = int(row.get("MoSold", 6))
        qual = int(row.get("OverallQual", 5))
        cond = int(row.get("OverallCond", 5))
        area = float(row.get("GrLivArea", 1500.0))

        formal_remod = (yr_remod > yr_built) and (yr_remod <= yr_sold)
        cosmetic_older = (not formal_remod) and ((yr_sold - yr_built) > 20) and (rng.random() < 0.25)
        has_reno = 1 if (formal_remod or cosmetic_older) else 0

        if has_reno == 1:
            if qual >= 8 and formal_remod:
                reno_type = str(rng.choice(["Major_Structural", "Kitchen_Bath"], p=[0.45, 0.55]))
            elif qual >= 6 and formal_remod:
                reno_type = str(rng.choice(["Kitchen_Bath", "Cosmetic"], p=[0.65, 0.35]))
            else:
                reno_type = "Cosmetic"

            base_cost = {"Cosmetic": 9500.0, "Kitchen_Bath": 26000.0, "Major_Structural": 52000.0}[reno_type]
            size_factor = max(0.75, min(1.60, area / 1500.0))
            reno_cost = float(np.clip(round(base_cost * size_factor * rng.lognormal(0.0, 0.25), 2), 2500.0, 125000.0))
            contractor = str(contractor_pool[int(rng.integers(0, len(contractor_pool)))])
            permit_yr = min(max(yr_remod, yr_built), yr_sold)
            permit_id = f"PRM-{permit_yr}-{int(rng.integers(10000, 99999))}"
        else:
            reno_type = "None"
            reno_cost = 0.0
            contractor = "None"
            permit_id = "None"

        # 3. Days on Market (DOM)
        dom_base = 42.0 * math.exp(-0.07 * (qual - 5) - 0.04 * (cond - 5))
        season_mult = 0.85 if mo_sold in [4, 5, 6, 7] else (1.18 if mo_sold in [11, 12, 1, 2] else 1.0)
        dom = int(np.clip(round(dom_base * season_mult * rng.lognormal(0.0, 0.45)), 3, 365))

        # 4. Macroeconomic indicators
        macro = MACRO_BY_YEAR.get(yr_sold, {"rate": 5.80, "hpi": 98.0, "unemp": 4.8})
        seasonal_rate_adj = -0.08 * math.sin((mo_sold - 1) * math.pi / 6.0)
        mortgage_rate = float(np.clip(round(macro["rate"] + seasonal_rate_adj + rng.normal(0.0, 0.04), 2), 4.10, 6.90))
        hpi = float(np.clip(round(macro["hpi"] + 0.6 * math.sin((mo_sold - 3) * math.pi / 6.0) + rng.normal(0.0, 0.35), 2), 90.0, 105.5))
        unemp = float(np.clip(round(macro["unemp"] + rng.normal(0.0, 0.12), 2), 3.4, 6.7))

        records.append({
            "distance_to_school_km": round(d_school, 3),
            "distance_to_hospital_km": round(d_hosp, 3),
            "distance_to_transit_km": round(d_trans, 3),
            "amenity_composite_score": amenity_score,
            "has_renovation": has_reno,
            "renovation_type": reno_type,
            "renovation_cost_usd": reno_cost,
            "renovation_contractor": contractor,
            "renovation_permit_id": permit_id,
            "days_on_market": dom,
            "mortgage_rate_pct": mortgage_rate,
            "local_price_index": hpi,
            "unemployment_rate_pct": unemp,
        })

    synth_df = pd.DataFrame(records, index=df.index)
    return pd.concat([df, synth_df], axis=1)


def validate_synthetic_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Runs integrity and range sanity checks on the augmented dataset."""
    missing = int(df[SYNTHETIC_COLUMNS].isna().sum().sum())
    assert missing == 0, f"Found {missing} missing values in synthetic columns!"
    assert (df["distance_to_school_km"].between(0.20, 5.00)).all(), "School distance out of range"
    assert (df["distance_to_hospital_km"].between(0.50, 8.00)).all(), "Hospital distance out of range"
    assert (df["distance_to_transit_km"].between(0.08, 2.50)).all(), "Transit distance out of range"
    assert (df["amenity_composite_score"].between(25.0, 98.0)).all(), "Amenity score out of range"
    assert (df["days_on_market"].between(3, 365)).all(), "Days on market out of range"
    assert (df["renovation_cost_usd"] >= 0.0).all(), "Negative renovation cost found"

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_synthetic": missing,
        "renovation_rate_pct": round(float(df["has_renovation"].mean() * 100), 2),
        "avg_dom": round(float(df["days_on_market"].mean()), 2),
        "avg_amenity_score": round(float(df["amenity_composite_score"].mean()), 2),
    }


def run_module_1(data_dir: str = "data", output_dir: str = "data/processed", seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executes Module 1 synthetic data generation and validation for train and test splits."""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at {train_path}. Place Kaggle train.csv in {data_dir}/.")

    logger.info("Loading raw Kaggle datasets...")
    train_raw = pd.read_csv(train_path)
    train_aug = generate_contextual_features(train_raw, seed=seed)
    train_stats = validate_synthetic_data(train_aug)
    train_out = os.path.join(output_dir, "train_augmented.csv")
    train_aug.to_csv(train_out, index=False)
    logger.info(f"Saved {train_out} | Validation stats: {train_stats}")

    test_aug = pd.DataFrame()
    if os.path.exists(test_path):
        test_raw = pd.read_csv(test_path)
        test_aug = generate_contextual_features(test_raw, seed=seed + 1)
        test_stats = validate_synthetic_data(test_aug)
        test_out = os.path.join(output_dir, "test_augmented.csv")
        test_aug.to_csv(test_out, index=False)
        logger.info(f"Saved {test_out} | Validation stats: {test_stats}")

    return train_aug, test_aug


if __name__ == "__main__":
    run_module_1()