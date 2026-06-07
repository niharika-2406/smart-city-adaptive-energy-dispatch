import pandas as pd
import numpy as np

FILE = r"C:\Users\Admin\Desktop\EDI III\backend\data\processed\SMART_CITY_ADAPTIVE_DISPATCH.csv"

df = pd.read_csv(FILE)

# 1. Ensure solar_generation exists
if "solar_generation" not in df.columns:
    df["solar_generation"] = 0

# 2. Fix total_load
df["total_load"] = pd.to_numeric(df["total_load"], errors="coerce")

# 3. Recompute net_load PHYSICALLY
df["net_load"] = df["total_load"] - df["solar_generation"]

# 4. Fix adaptive_net_load
df["adaptive_net_load"] = df["net_load"]

# 5. Recompute adaptive outputs
df["adaptive_storage"] = (df["adaptive_net_load"] * 0.05).round(2)
df["adaptive_shedding"] = (df["adaptive_net_load"] * 0.02).round(2)

# 6. Recompute adaptive stress correctly
df["adaptive_stress"] = (
    df["adaptive_net_load"] / df["adaptive_net_load"].max()
) * 100

# 7. Final safety cleanup
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# 8. Save
df.to_csv(FILE, index=False)

print("✅ DATASET FULLY REPAIRED")
print(df.tail(1)[[
    "total_load",
    "solar_generation",
    "net_load",
    "adaptive_net_load",
    "adaptive_storage",
    "adaptive_shedding",
    "adaptive_stress"
]])
