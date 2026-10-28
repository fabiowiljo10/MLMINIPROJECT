"""
generate_dataset.py — Download & Indianize the Kaggle Medical Cost Dataset
============================================================================
Downloads the real Kaggle Medical Cost Personal Dataset (1,338 records) from
GitHub and adapts it for the Indian context:
  - US regions → Indian regions (north, south, east, west, central, northeast)
  - USD charges → INR charges (with regional cost-of-living multipliers)
  - Saves as insurance.csv ready for training
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ── 1. Load the REAL Kaggle Medical Cost dataset ───────────────────────
KAGGLE_URL = (
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets"
    "/master/insurance.csv"
)

print("[INFO] Downloading real Kaggle Medical Cost Personal Dataset ...")
try:
    df = pd.read_csv(KAGGLE_URL)
    print(f"[OK]  Downloaded {len(df)} rows from GitHub.")
except Exception as e:
    raise RuntimeError(
        f"Could not download dataset from:\n  {KAGGLE_URL}\n"
        "Check your internet connection.\n"
        f"Error: {e}"
    )

# ── 2. Map US regions → Indian regions ─────────────────────────────────
# The original dataset has 4 US regions: northeast, southeast, southwest, northwest
# We map them to 6 Indian regions with realistic insurance cost adjustments.

INDIAN_REGIONS = ["north", "south", "east", "west", "central", "northeast"]

# Deterministic mapping: spread 4 US regions across 6 Indian regions
# by assigning rows based on their original region + index position
REGION_MAP = {
    "northeast": ["north", "northeast"],     # US NE → India North / NE
    "southeast": ["south", "east"],          # US SE → India South / East
    "southwest": ["west", "central"],        # US SW → India West / Central
    "northwest": ["north", "south"],         # US NW → India North / South
}

new_regions = []
for idx, row in df.iterrows():
    choices = REGION_MAP[row["region"]]
    new_regions.append(choices[idx % len(choices)])

df["region"] = new_regions

# ── 3. Convert USD charges → INR charges ───────────────────────────────
# Base conversion: 1 USD ≈ 83 INR (approximate rate)
# But Indian insurance premiums are generally lower due to cost-of-living,
# so we apply a realistic scaling factor of ~0.35 to represent actual
# Indian medical insurance costs (₹25,000 – ₹15,00,000 range).

BASE_USD_TO_INR = 83.0
INDIA_COST_FACTOR = 0.35  # Indian medical costs are lower than US

# Regional cost-of-living multipliers for India
REGIONAL_MULTIPLIER = {
    "north":     1.05,   # Delhi NCR, Chandigarh — slightly expensive
    "south":     1.00,   # Bangalore, Chennai, Hyderabad — baseline
    "east":      0.88,   # Kolkata, Bhubaneswar — lower cost
    "west":      1.12,   # Mumbai, Pune — most expensive
    "central":   0.85,   # Bhopal, Nagpur — lowest cost
    "northeast": 0.82,   # Guwahati, Shillong — lowest cost
}

# Apply conversion with regional adjustment and add realistic noise
charges_inr = []
for idx, row in df.iterrows():
    base_inr = row["charges"] * BASE_USD_TO_INR * INDIA_COST_FACTOR
    regional_adj = REGIONAL_MULTIPLIER[row["region"]]
    noise = np.random.normal(1.0, 0.03)  # ±3% random variation
    final = base_inr * regional_adj * noise
    charges_inr.append(max(15000.0, round(final, 2)))  # Min ₹15,000

df["charges"] = charges_inr

# ── 4. Save the Indianized dataset ────────────────────────────────────
df.to_csv("insurance.csv", index=False)
print(f"\n[OK] Generated insurance.csv  ({len(df)} rows)")
print(f"     Regions: {sorted(df['region'].unique())}")
print(f"     Charges range: INR {df['charges'].min():,.0f} - INR {df['charges'].max():,.0f}")
print(f"     Mean charge: INR {df['charges'].mean():,.0f}")
print()
print(df.describe().round(2))
print("\nSample rows:")
print(df.head(10))
