import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ==============================
# FILE PATHS
# ==============================
FORECAST_FILE = "backend/data/forecasts/timegpt_forecast.csv"
OPT_FILE = "backend/data/processed/SMART_CITY_OPTIMIZED_DISPATCH.csv"
OUT_FILE = "backend/data/processed/SMART_CITY_ADAPTIVE_DISPATCH.csv"
FIG_DIR = "reports/adaptive"

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
df_opt = pd.read_csv(OPT_FILE, parse_dates=["datetime"])
df_fc  = pd.read_csv(FORECAST_FILE, parse_dates=["datetime"])

df = df_opt.merge(df_fc, on="datetime", how="left")

print("✅ Columns available for Adaptive AI:")
print(df.columns.tolist())

# ==============================
# FORECAST COLUMN AUTO-DETECTION
# ==============================
if "forecast" in df.columns:
    base_forecast_col = "forecast"
elif "predicted_load" in df.columns:
    base_forecast_col = "predicted_load"
elif "net_load" in df.columns:
    base_forecast_col = "net_load"
elif "total_load" in df.columns:
    base_forecast_col = "total_load"
else:
    raise ValueError("❌ No valid forecast column found for risk computation.")

print(f"✅ Forecast column detected: {base_forecast_col}")

# ==============================
# UNCERTAINTY INDEX
# ==============================
if "upper" in df.columns and "lower" in df.columns:
    # Case 1: TimeGPT prediction interval exists
    df["forecast_uncertainty"] = df["upper"] - df["lower"]
    print("✅ Using TimeGPT prediction interval for uncertainty")
else:
    # Case 2: No PI available → synthesize from rolling volatility
    df["forecast_uncertainty"] = (
        df[base_forecast_col].rolling(6).std().fillna(10)
    )
    print("⚠ No PI found — using rolling volatility uncertainty")

# ==============================
# RISK INDEX (SAFE)
# ==============================
df["risk_index"] = df["forecast_uncertainty"] / df[base_forecast_col].replace(0, np.nan)
df["risk_index"] = df["risk_index"].fillna(0.1)

# ==============================
# ADAPTIVE BATTERY & SHEDDING
# ==============================
df["adaptive_storage"] = 0.0
df["adaptive_shedding"] = 0.0

for i, row in df.iterrows():

    base_storage = row["storage_available"]
    net_load = row["net_load"]
    risk = row["risk_index"]

    # High uncertainty → charge more reserve
    storage = base_storage * (1 + 0.7 * risk)

    # Shedding only when BOTH system stress & uncertainty are high
    if net_load > row["res_capacity"] * 0.95 and risk > 0.18:
        shed = 0.12 * net_load
    else:
        shed = 0.0

    df.loc[i, "adaptive_storage"] = storage
    df.loc[i, "adaptive_shedding"] = shed

# ==============================
# FINAL LOAD AFTER ADAPTIVE CONTROL
# ==============================
df["adaptive_net_load"] = (
    df["net_load"] - df["adaptive_shedding"] - df["adaptive_storage"]
)

df["adaptive_stress"] = df["adaptive_net_load"] / (
    df["res_capacity"] + df["com_capacity"]
)

# ==============================
# SAVE FINAL RESULT
# ==============================
df.to_csv(OUT_FILE, index=False)
print("✅ Adaptive AI dispatch generated:", OUT_FILE)

# ==============================
# VISUAL VALIDATION
# ==============================
plt.figure()
plt.plot(df["stress_res"], label="Static Stress")
plt.plot(df["adaptive_stress"], label="Adaptive Stress")
plt.legend()
plt.title("Static vs Adaptive Grid Stress")
plt.savefig(f"{FIG_DIR}/adaptive_vs_static_stress.png")
plt.close()

plt.figure()
plt.plot(df["adaptive_storage"])
plt.title("Battery Storage Response to Risk")
plt.savefig(f"{FIG_DIR}/storage_response.png")
plt.close()

plt.figure()
plt.plot(df["adaptive_shedding"])
plt.title("Adaptive Load Shedding")
plt.savefig(f"{FIG_DIR}/shedding_response.png")
plt.close()

print("✅ Adaptive control figures saved.")
