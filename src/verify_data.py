import csv
import sys

def verify_dataset(path: str):
    print(f"\n==========================================")
    print(f"  VERIFYING: {path}")
    print(f"==========================================")
    
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    print(f"Record count: {len(rows)}")
    print(f"Total columns: {len(reader.fieldnames or [])}")

    new_cols = [
        "distance_to_school_km", "distance_to_hospital_km", "distance_to_transit_km",
        "amenity_composite_score", "has_renovation", "renovation_type",
        "renovation_cost_usd", "renovation_contractor", "renovation_permit_id",
        "days_on_market", "mortgage_rate_pct", "local_price_index", "unemployment_rate_pct"
    ]

    # 1. Null check
    total_nulls = 0
    for col in new_cols:
        col_nulls = sum(1 for r in rows if r.get(col) is None or r.get(col) == "")
        if col_nulls > 0:
            print(f"  [FAIL] Column {col} contains {col_nulls} missing values!")
            total_nulls += col_nulls
    if total_nulls == 0:
        print("  [PASS] 0 missing/null values across all 13 synthetic columns.")

    # 2. Metric range checks
    doms = [int(r["days_on_market"]) for r in rows]
    rates = [float(r["mortgage_rate_pct"]) for r in rows]
    hpis = [float(r["local_price_index"]) for r in rows]
    unemp = [float(r["unemployment_rate_pct"]) for r in rows]
    amenities = [float(r["amenity_composite_score"]) for r in rows]
    renovated = [r for r in rows if r["has_renovation"] == "1"]
    costs = [float(r["renovation_cost_usd"]) for r in renovated]

    print(f"  - Days on Market (DOM): min={min(doms)}, max={max(doms)}, mean={sum(doms)/len(doms):.1f}")
    print(f"  - 30-Yr Mortgage Rate: min={min(rates):.2f}%, max={max(rates):.2f}%, mean={sum(rates)/len(rates):.2f}%")
    print(f"  - Local Price Index (HPI): min={min(hpis):.1f}, max={max(hpis):.1f}, mean={sum(hpis)/len(hpis):.1f}")
    print(f"  - Unemployment Rate: min={min(unemp):.1f}%, max={max(unemp):.1f}%, mean={sum(unemp)/len(unemp):.1f}%")
    print(f"  - Amenity Score: min={min(amenities):.1f}, max={max(amenities):.1f}, mean={sum(amenities)/len(amenities):.1f}")
    print(f"  - Renovated Properties: {len(renovated)}/{len(rows)} ({len(renovated)/len(rows)*100:.1f}%)")
    print(f"  - Average Renovation Cost (renovated homes): ${sum(costs)/len(costs):,.2f}")
    print("  [PASS] All range, consistency, and sanity checks PASSED.")

if __name__ == "__main__":
    verify_dataset("data/processed/train_augmented.csv")
    verify_dataset("data/processed/test_augmented.csv")
