import os
import numpy as np
import pandas as pd

# ======================================================
# CONFIG
# ======================================================
# Run this from the project ROOT:
#   python scripts/optimize_power_dispatch.py
#
# It will read your final dataset and create:
#   data/processed/SMART_CITY_OPTIMIZED_DISPATCH.csv

INPUT_CSV = "data/proccessed/FINAL_SMART_CITY_DATASET_3YEAR.csv"
OUTPUT_CSV = "data/proccessed/SMART_CITY_OPTIMIZED_DISPATCH.csv"

# Name of the column that has the demand we want to serve.
# For now we use the actual historical "total_load".
# Later you can swap this to a forecast column if you add one.
DEMAND_COLUMN = "total_load"   # change to "xgb_pred" etc if needed

# ------------------------------------------------------
# Helper: make sure a column exists, otherwise create with default
# ------------------------------------------------------
def ensure_column(df, col, default_value):
    if col not in df.columns:
        print(f"[WARN] Column '{col}' not found, creating with default={default_value}")
        df[col] = default_value
    return df


# ======================================================
# STEP 1: LOAD DATA
# ======================================================
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input dataset not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
print(f"[INFO] Loaded dataset: {INPUT_CSV}  (rows={len(df)}, cols={len(df.columns)})")

if DEMAND_COLUMN not in df.columns:
    raise ValueError(
        f"Column '{DEMAND_COLUMN}' not found in dataset. "
        f"Available columns: {list(df.columns)[:20]} ..."
    )

# ======================================================
# STEP 2: ENSURE REQUIRED COLUMNS EXIST
# ======================================================

# Zone weights (how much of demand belongs to each zone)
df = ensure_column(df, "residential_weight", 0.5)
df = ensure_column(df, "commercial_weight", 0.3)
df = ensure_column(df, "event_weight", 0.2)

# Normalize weights so they sum to 1
weight_sum = (
    df["residential_weight"] + df["commercial_weight"] + df["event_weight"]
)
df["residential_weight"] /= weight_sum
df["commercial_weight"] /= weight_sum
df["event_weight"] /= weight_sum

# Zone capacities (maximum safe supply for each zone)
# If not present, we approximate from the maximum historical demand.
max_demand = df[DEMAND_COLUMN].max()

df = ensure_column(df, "res_capacity", 0.45 * max_demand)
df = ensure_column(df, "com_capacity", 0.35 * max_demand)
df = ensure_column(df, "event_capacity", 0.25 * max_demand)

# Storage & flexibility
df = ensure_column(df, "storage_available", 0.08 * max_demand)   # MWh-ish
df = ensure_column(df, "flexible_load_ratio", 0.2)               # 20% can shift/shed


# ======================================================
# STEP 3: ROW-WISE "OPTIMIZATION" (GREEDY DISPATCH)
# ======================================================
# NOTE:
#   This is a rule-based optimization:
#   - Minimize load shedding
#   - Protect residential first, then commercial, then events
#   - Use storage to cover shortages, up to storage_available
#
#   It is simple but explainable and works well for your project scale.

def optimize_row(row):
    demand = row[DEMAND_COLUMN]

    # 1) Compute zone demand
    res_d = demand * row["residential_weight"]
    com_d = demand * row["commercial_weight"]
    evt_d = demand * row["event_weight"]

    # 2) Zone capacities
    res_cap = row["res_capacity"]
    com_cap = row["com_capacity"]
    evt_cap = row["event_capacity"]

    # 3) Initial supply = min(demand, capacity)
    res_sup = min(res_d, res_cap)
    com_sup = min(com_d, com_cap)
    evt_sup = min(evt_d, evt_cap)

    # 4) Initial shedding (before storage / shifting)
    shed_res = max(0.0, res_d - res_sup)
    shed_com = max(0.0, com_d - com_sup)
    shed_evt = max(0.0, evt_d - evt_sup)

    total_shed = shed_res + shed_com + shed_evt

    # 5) Bound shedding by flexible_load_ratio (cannot shed all load)
    flex = float(row["flexible_load_ratio"])
    max_shed_res = flex * res_d
    max_shed_com = flex * com_d
    max_shed_evt = flex * evt_d

    shed_res = min(shed_res, max_shed_res)
    shed_com = min(shed_com, max_shed_com)
    shed_evt = min(shed_evt, max_shed_evt)
    total_shed = shed_res + shed_com + shed_evt

    # 6) Use storage to reduce shedding, prioritising:
    #    1) Residential  2) Commercial  3) Events
    storage = float(row["storage_available"])
    storage_used = min(storage, total_shed)
    remaining = storage_used

    # reduce residential shedding first
    reduce_res = min(shed_res, remaining)
    shed_res -= reduce_res
    remaining -= reduce_res

    # then commercial
    reduce_com = min(shed_com, remaining)
    shed_com -= reduce_com
    remaining -= reduce_com

    # then events
    reduce_evt = min(shed_evt, remaining)
    shed_evt -= reduce_evt
    remaining -= reduce_evt

    # 7) Recompute final supplies after storage support
    res_sup = res_d - shed_res
    com_sup = com_d - shed_com
    evt_sup = evt_d - shed_evt

    # 8) Compute post-optimization stress ratios
    #    (how close each zone is to its capacity)
    stress_res = res_sup / res_cap if res_cap > 0 else 0.0
    stress_com = com_sup / com_cap if com_cap > 0 else 0.0
    stress_evt = evt_sup / evt_cap if evt_cap > 0 else 0.0

    # 9) Simple overload flag
    overload_flag = int(stress_res > 1.0 or stress_com > 1.0 or stress_evt > 1.0)

    return pd.Series(
        {
            # Demands
            "res_demand": res_d,
            "com_demand": com_d,
            "event_demand": evt_d,

            # Supplies after optimization
            "res_supply": res_sup,
            "com_supply": com_sup,
            "event_supply": evt_sup,

            # Final shedding (after storage)
            "shed_res": shed_res,
            "shed_com": shed_com,
            "shed_evt": shed_evt,
            "total_shed_after": shed_res + shed_com + shed_evt,

            # Storage used
            "storage_used": storage_used,

            # Stress after optimization
            "stress_res_after": stress_res,
            "stress_com_after": stress_com,
            "stress_evt_after": stress_evt,

            # Grid overload indicator
            "grid_overload_flag": overload_flag,
        }
    )


print("[INFO] Running greedy optimization for each hour...")
opt_results = df.apply(optimize_row, axis=1)

# Attach results to the main dataframe
df_opt = pd.concat([df, opt_results], axis=1)

# ======================================================
# STEP 4: SAVE RESULT
# ======================================================
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_opt.to_csv(OUTPUT_CSV, index=False)

print("[INFO] Optimization complete.")
print(f"[INFO] Saved optimized dispatch dataset → {OUTPUT_CSV}")
print(f"[INFO] Final shape: {df_opt.shape[0]} rows x {df_opt.shape[1]} columns")
