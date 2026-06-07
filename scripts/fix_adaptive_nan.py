import pandas as pd

df = pd.read_csv("C:/Users/Admin/Desktop/EDI III/backend/data/processed/SMART_CITY_ADAPTIVE_DISPATCH.csv")

# Fill adaptive_storage using 5% of net load
df["adaptive_storage"] = df["adaptive_net_load"] * 0.05

# Fill adaptive_shedding using 2% of net load
df["adaptive_shedding"] = df["adaptive_net_load"] * 0.02

# Fill adaptive_stress using normalized risk index
df["adaptive_stress"] = df["risk_index"] * 100

# Safety fill
df.fillna(0, inplace=True)

df.to_csv("C:/Users/Admin/Desktop/EDI III/backend/data/processed/Adaptive_NaNs_FIXED.csv", index=False)

print("✅ Adaptive NaNs FIXED & SAVED")
