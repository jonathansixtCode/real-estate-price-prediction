"""
Module 1: Synthetic Contextual Data Generation
Real Estate House Price Prediction Project

This module enriches the Ames Housing dataset with synthetic contextual features
as required by the project specifications:
1. Neighborhood Amenity Scores (distance to school, hospital, transit + composite score)
2. Renovation History (renovation type, estimated cost, contractor, permit ID)
3. Days-on-Market (realistic listing duration based on property quality and seasonality)
4. Macro Indicators (historical mortgage interest rate, local price index, unemployment rate)

Designed with zero required external dependencies (runs with standard library csv/math/random),
while providing seamless integration with pandas and faker if they are installed.
Deterministic and bit-for-bit reproducible via a fixed random seed.
"""

import os
import sys
import csv
import math
import random
import logging
from typing import Dict, Any, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("DataGeneration")

# Optional imports for enhanced features if installed
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False


# -----------------------------------------------------------------------------
# 1. Neighborhood Amenity Baselines for Ames, Iowa
# -----------------------------------------------------------------------------
# Realistic average distances (in kilometers) from neighborhood centers to:
# - School (Ames High / Middle / Elementary schools)
# - Hospital (Mary Greeley Medical Center, central Ames)
# - Transit (CyRide primary transit hub)
NEIGHBORHOOD_AMENITIES_BASELINE: Dict[str, Dict[str, float]] = {
    # Central / Downtown / University Adjacent
    "OldTown": {"school": 0.6, "hospital": 0.8, "transit": 0.25},
    "BrkSide": {"school": 0.8, "hospital": 1.1, "transit": 0.30},
    "SWISU":   {"school": 1.2, "hospital": 3.2, "transit": 0.20},
    "Crawfor": {"school": 1.4, "hospital": 2.5, "transit": 0.40},
    "IDOTRR":  {"school": 0.9, "hospital": 1.5, "transit": 0.30},
    "Edwards": {"school": 1.1, "hospital": 3.0, "transit": 0.35},
    "CollgCr": {"school": 1.3, "hospital": 4.2, "transit": 0.40},
    # North Ames & Suburbs
    "NAmes":   {"school": 0.7, "hospital": 2.2, "transit": 0.35},
    "NWAmes":  {"school": 1.2, "hospital": 3.8, "transit": 0.50},
    "Gilbert": {"school": 2.1, "hospital": 5.5, "transit": 0.80},
    "Sawyer":  {"school": 1.0, "hospital": 3.5, "transit": 0.45},
    "SawyerW": {"school": 1.4, "hospital": 4.5, "transit": 0.55},
    "Somerst": {"school": 1.5, "hospital": 3.4, "transit": 0.40},
    "StoneBr": {"school": 2.2, "hospital": 4.0, "transit": 0.70},
    "NridgHt": {"school": 2.4, "hospital": 5.1, "transit": 0.75},
    "NoRidge": {"school": 2.6, "hospital": 5.4, "transit": 0.80},
    "Veenker": {"school": 1.8, "hospital": 3.1, "transit": 0.50},
    # Outer / Rural / PUD
    "Mitchel": {"school": 2.8, "hospital": 5.8, "transit": 0.90},
    "Timber":  {"school": 3.5, "hospital": 6.8, "transit": 1.20},
    "ClearCr": {"school": 3.2, "hospital": 6.0, "transit": 1.10},
    "Blmngtn": {"school": 1.9, "hospital": 3.6, "transit": 0.60},
    "BrDale":  {"school": 1.1, "hospital": 2.8, "transit": 0.40},
    "NPkVill": {"school": 1.3, "hospital": 2.9, "transit": 0.45},
    "Blueste": {"school": 2.0, "hospital": 4.2, "transit": 0.65},
    "MeadowV": {"school": 2.7, "hospital": 5.2, "transit": 0.85},
}

DEFAULT_AMENITY_BASELINE = {"school": 1.8, "hospital": 3.8, "transit": 0.50}

REGIONAL_CONTRACTORS = [
    "Heartland Home Improvements LLC",
    "Ames Heritage Remodeling",
    "Midwest Craft Builders Inc.",
    "Prairie State Renovations",
    "Cyclone City Contractors",
    "Iowa Modern Living Remodeling",
    "Oak & Timber Carpentry Co.",
    "Apex Residential Builders",
    "Story County Design & Build",
    "Pinnacle Renovation Services",
    "TrueNorth Home Restorations",
    "Hawkeye Elite Remodeling",
    "Cedar Valley Restoration Pros",
    "Prairie Ridge Home Craftsmen"
]


# -----------------------------------------------------------------------------
# 2. Historical Macroeconomic Data (US / Iowa 2006-2010)
# -----------------------------------------------------------------------------
HISTORICAL_MORTGAGE_RATES: Dict[Tuple[int, int], float] = {
    (2006, 1): 6.15, (2006, 2): 6.25, (2006, 3): 6.32, (2006, 4): 6.51,
    (2006, 5): 6.60, (2006, 6): 6.68, (2006, 7): 6.76, (2006, 8): 6.52,
    (2006, 9): 6.40, (2006, 10): 6.36, (2006, 11): 6.24, (2006, 12): 6.14,

    (2007, 1): 6.22, (2007, 2): 6.34, (2007, 3): 6.16, (2007, 4): 6.18,
    (2007, 5): 6.26, (2007, 6): 6.66, (2007, 7): 6.70, (2007, 8): 6.57,
    (2007, 9): 6.38, (2007, 10): 6.37, (2007, 11): 6.21, (2007, 12): 6.10,

    (2008, 1): 5.76, (2008, 2): 5.92, (2008, 3): 5.97, (2008, 4): 5.88,
    (2008, 5): 6.04, (2008, 6): 6.32, (2008, 7): 6.43, (2008, 8): 6.48,
    (2008, 9): 6.11, (2008, 10): 6.20, (2008, 11): 6.09, (2008, 12): 5.33,

    (2009, 1): 5.05, (2009, 2): 5.13, (2009, 3): 5.00, (2009, 4): 4.81,
    (2009, 5): 4.86, (2009, 6): 5.42, (2009, 7): 5.22, (2009, 8): 5.19,
    (2009, 9): 5.06, (2009, 10): 4.95, (2009, 11): 4.78, (2009, 12): 4.93,

    (2010, 1): 5.03, (2010, 2): 4.99, (2010, 3): 4.97, (2010, 4): 5.10,
    (2010, 5): 4.89, (2010, 6): 4.74, (2010, 7): 4.57, (2010, 8): 4.43,
    (2010, 9): 4.35, (2010, 10): 4.23, (2010, 11): 4.30, (2010, 12): 4.71,
}

HISTORICAL_HPI: Dict[Tuple[int, int], float] = {
    (2006, 1): 100.0, (2006, 4): 101.4, (2006, 7): 102.8, (2006, 10): 103.2,
    (2007, 1): 103.5, (2007, 4): 104.1, (2007, 7): 103.7, (2007, 10): 102.3,
    (2008, 1): 101.5, (2008, 4): 100.2, (2008, 7): 98.4,  (2008, 10): 95.8,
    (2009, 1): 94.2,  (2009, 4): 93.1,  (2009, 7): 92.5,  (2009, 10): 91.8,
    (2010, 1): 91.2,  (2010, 4): 92.0,  (2010, 7): 92.8,  (2010, 10): 92.4,
}

HISTORICAL_UNEMPLOYMENT: Dict[int, float] = {
    2006: 3.7,
    2007: 3.9,
    2008: 4.2,
    2009: 6.3,
    2010: 6.0,
}

NEW_FEATURE_COLUMNS = [
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
    "unemployment_rate_pct"
]


# -----------------------------------------------------------------------------
# 3. Core Row-Level Generation Logic
# -----------------------------------------------------------------------------

def generate_record_features(
    record: Dict[str, Any],
    rng: random.Random,
    faker_instance: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Computes all 13 synthetic contextual features for a single property record.
    """
    # Safe type extraction with defaults
    neigh = record.get("Neighborhood", "")
    try:
        yr_built = int(record.get("YearBuilt", 1970))
    except (ValueError, TypeError):
        yr_built = 1970

    try:
        yr_remod = int(record.get("YearRemodAdd", yr_built))
    except (ValueError, TypeError):
        yr_remod = yr_built

    try:
        yr_sold = int(record.get("YrSold", 2008))
    except (ValueError, TypeError):
        yr_sold = 2008

    try:
        mo_sold = int(record.get("MoSold", 6))
    except (ValueError, TypeError):
        mo_sold = 6

    try:
        qual = float(record.get("OverallQual", 5))
    except (ValueError, TypeError):
        qual = 5.0

    try:
        cond = float(record.get("OverallCond", 5))
    except (ValueError, TypeError):
        cond = 5.0

    try:
        gr_liv_area = float(record.get("GrLivArea", 1500))
    except (ValueError, TypeError):
        gr_liv_area = 1500.0

    # 1. Amenity Features
    baseline = NEIGHBORHOOD_AMENITIES_BASELINE.get(neigh, DEFAULT_AMENITY_BASELINE)
    d_school = max(0.20, round(baseline["school"] + rng.gauss(0, 0.12), 2))
    d_hospital = max(0.50, round(baseline["hospital"] + rng.gauss(0, 0.18), 2))
    d_transit = max(0.08, round(baseline["transit"] + rng.gauss(0, 0.08), 2))

    comp_raw = 100.0 - (d_school * 8.0 + d_hospital * 4.5 + d_transit * 25.0) + rng.gauss(0, 2.0)
    amenity_score = round(max(25.0, min(98.0, comp_raw)), 1)

    # 2. Renovation Features
    was_remod = (yr_remod > yr_built) and (yr_remod <= yr_sold)
    age_at_sale = max(0, yr_sold - yr_built)
    unrecorded_update = (not was_remod) and (age_at_sale > 20) and (rng.random() < 0.25)

    if was_remod or unrecorded_update:
        has_renov = 1
        p_type = rng.random()
        if qual >= 8:
            if p_type < 0.40:
                r_type = "Major_Structural"
            elif p_type < 0.85:
                r_type = "Kitchen_Bath"
            else:
                r_type = "Cosmetic"
        elif qual <= 4:
            if p_type < 0.65:
                r_type = "Cosmetic"
            elif p_type < 0.90:
                r_type = "Kitchen_Bath"
            else:
                r_type = "Major_Structural"
        else:
            if p_type < 0.35:
                r_type = "Cosmetic"
            elif p_type < 0.75:
                r_type = "Kitchen_Bath"
            else:
                r_type = "Major_Structural"

        area_factor = gr_liv_area / 1500.0
        if r_type == "Cosmetic":
            cost = rng.lognormvariate(math.log(6500), 0.35) * area_factor
        elif r_type == "Kitchen_Bath":
            cost = rng.lognormvariate(math.log(24000), 0.40) * area_factor
        else:
            cost = rng.lognormvariate(math.log(58000), 0.45) * area_factor

        renov_cost = round(cost, 2)
        if faker_instance:
            contractor = faker_instance.company() + " Remodeling"
            permit_id = f"PRM-{yr_remod}-{faker_instance.bothify(text='??#####').upper()}"
        else:
            contractor = rng.choice(REGIONAL_CONTRACTORS)
            permit_num = rng.randint(10000, 99999)
            permit_id = f"PRM-{yr_remod}-IA{permit_num}"
    else:
        has_renov = 0
        r_type = "None"
        renov_cost = 0.0
        contractor = "None"
        permit_id = "None"

    # 3. Days on Market (DOM)
    mu = 3.74  # median ~ 42 days
    qual_adj = -0.09 * (qual - 5.5) - 0.05 * (cond - 5.5)
    if mo_sold in [4, 5, 6, 7]:
        season_adj = -0.18
    elif mo_sold in [11, 12, 1, 2]:
        season_adj = 0.22
    else:
        season_adj = 0.0

    log_dom = rng.gauss(mu + qual_adj + season_adj, 0.48)
    dom = int(round(math.exp(log_dom)))
    dom = max(3, min(365, dom))

    # 4. Macro Indicators
    base_rate = HISTORICAL_MORTGAGE_RATES.get((yr_sold, mo_sold), 5.50)
    mortgage_rate = round(base_rate + rng.gauss(0, 0.04), 2)

    quarter_mo = ((mo_sold - 1) // 3) * 3 + 1
    base_hpi = HISTORICAL_HPI.get((yr_sold, quarter_mo), 96.0)
    hpi = round(base_hpi + rng.gauss(0, 0.3), 1)

    base_unemp = HISTORICAL_UNEMPLOYMENT.get(yr_sold, 5.0)
    unemp = round(base_unemp + rng.gauss(0, 0.1), 1)

    return {
        "distance_to_school_km": d_school,
        "distance_to_hospital_km": d_hospital,
        "distance_to_transit_km": d_transit,
        "amenity_composite_score": amenity_score,
        "has_renovation": has_renov,
        "renovation_type": r_type,
        "renovation_cost_usd": renov_cost,
        "renovation_contractor": contractor,
        "renovation_permit_id": permit_id,
        "days_on_market": dom,
        "mortgage_rate_pct": mortgage_rate,
        "local_price_index": hpi,
        "unemployment_rate_pct": unemp
    }


# -----------------------------------------------------------------------------
# 4. Dataset Processing Pipeline (Self-Contained Standard CSV)
# -----------------------------------------------------------------------------

def augment_csv_file(
    input_path: str,
    output_path: str,
    seed: int = 42
) -> int:
    """
    Reads an Ames CSV file, augments each row with synthetic features,
    and writes the result to output_path.
    
    Returns the count of processed rows.
    """
    rng = random.Random(seed)
    faker_instance = None
    if FAKER_AVAILABLE:
        faker_instance = Faker("en_US")
        faker_instance.seed_instance(seed)

    logger.info(f"Reading input from {input_path}...")
    with open(input_path, mode="r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])
        
        # Add new synthetic feature column headers
        augmented_fields = fieldnames + NEW_FEATURE_COLUMNS
        
        rows_to_write = []
        for row in reader:
            synthetic_feats = generate_record_features(row, rng, faker_instance)
            row.update(synthetic_feats)
            rows_to_write.append(row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode="w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=augmented_fields)
        writer.writeheader()
        writer.writerows(rows_to_write)

    logger.info(f"Successfully wrote {len(rows_to_write)} augmented records with {len(augmented_fields)} columns to {output_path}")
    return len(rows_to_write)


# -----------------------------------------------------------------------------
# 5. CLI Execution Entry Point
# -----------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic contextual features for Ames Real Estate dataset.")
    parser.add_argument("--data-dir", default="data", help="Directory containing train.csv and test.csv")
    parser.add_argument("--output-dir", default="data/processed", help="Directory to save augmented datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    train_in = os.path.join(args.data_dir, "train.csv")
    test_in = os.path.join(args.data_dir, "test.csv")
    train_out = os.path.join(args.output_dir, "train_augmented.csv")
    test_out = os.path.join(args.output_dir, "test_augmented.csv")

    if not os.path.exists(train_in):
        logger.error(f"Input file not found: {train_in}")
        sys.exit(1)

    print("=" * 70)
    print("  MODULE 1: SYNTHETIC DATA GENERATION PIPELINE")
    print("=" * 70)

    # 1. Augment train dataset
    train_count = augment_csv_file(train_in, train_out, seed=args.seed)
    print(f"[OK] train_augmented.csv generated ({train_count} rows, {len(NEW_FEATURE_COLUMNS)} new synthetic features)")

    # 2. Augment test dataset
    if os.path.exists(test_in):
        # Use seed + 1 for test set to preserve independent pseudo-random stream
        test_count = augment_csv_file(test_in, test_out, seed=args.seed + 1)
        print(f"[OK] test_augmented.csv generated ({test_count} rows, {len(NEW_FEATURE_COLUMNS)} new synthetic features)")

    print("=" * 70)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
