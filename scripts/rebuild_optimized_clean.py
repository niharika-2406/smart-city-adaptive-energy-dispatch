import pandas as pd
import numpy as np
import os

INPUT = r"C:/Users/Admin/Desktop/EDI III/backend/data/processed/SMART_CITY_FINAL_DATASET.csv"
OUT   = r"C:/Users/Admin/Desktop/EDI III/backend/data/processed/SMART_CITY_OPTIMIZED_DISPATCH.csv"

df = pd.read_csv(INPUT)

# Force valid total load
df["total_load"] = pd.to_numeric(df["total_load"], errors="coerce")
df["total_load"].fillna(df["total_load"].mean(), inplace=True)

# Capacities
max_demand = df["total_load"].max()
df["res_capacity"]   = 0.45 * max_demand
df["com_capacity"]   = 0.35 * max_demand
df["event_capacity"] = 0.25 * max_demand

# Storage & flexibility
df["storage_available"] = 0.08 * max_demand
df["flexible_load_ratio"] = 0.2

# Simple net load (no solar here)
df["net_load"] = df["total_load"]

# Save clean optimized file
df.to_csv(OUT, index=False)
print("✅ CLEAN OPTIMIZED FILE GENERATED")
print(df[["total_load","net_load","storage_available"]].tail())
