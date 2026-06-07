import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================
# CONFIG
# ==============================
INPUT_FILE = "C:/Users/Admin/Desktop/EDI III/data/processed/SMART_CITY_OPTIMIZED_DISPATCH.csv"
OUTPUT_FILE = "C:/Users/Admin/Desktop/EDI III/data/processed/SMART_CITY_EXTREME_EVENTS.csv"
FIGURE_DIR = "C:/Users/Admin/Desktop/EDI III/reports/extreme"

os.makedirs(FIGURE_DIR, exist_ok=True)
np.random.seed(42)

# ==============================
# LOAD OPTIMIZED DATA
# ==============================
df = pd.read_csv(INPUT_FILE)

# ==============================
# EXTREME EVENT FLAGS
# ==============================
df["event_heatwave"] = 0
df["event_festival"] = 0
df["event_ev_spike"] = 0
df["event_capacity_loss"] = 0
df["event_solar_drop"] = 0

event_indices = df.sample(frac=0.03).index

for idx in event_indices:
    event_type = np.random.choice(
        ["heatwave", "festival", "ev", "capacity", "solar"]
    )

    if event_type == "heatwave":
        df.loc[idx, "event_heatwave"] = 1
        df.loc[idx, "total_load"] *= 1.25

    elif event_type == "festival":
        df.loc[idx, "event_festival"] = 1
        df.loc[idx, "event_load"] *= 1.50
        df.loc[idx, "total_load"] *= 1.30

    elif event_type == "ev":
        df.loc[idx, "event_ev_spike"] = 1
        df.loc[idx, "total_load"] *= 1.40

    elif event_type == "capacity":
        df.loc[idx, "event_capacity_loss"] = 1
        df.loc[idx, "res_capacity"] *= 0.85
        df.loc[idx, "com_capacity"] *= 0.85
        df.loc[idx, "event_capacity"] *= 0.85

    elif event_type == "solar":
        df.loc[idx, "event_solar_drop"] = 1
        df.loc[idx, "solar_generation"] *= 0.40

# ==============================
# EXTREME STRESS
# ==============================
df["stress_res_extreme"] = df["res_load_after"] / df["res_capacity"]
df["stress_com_extreme"] = df["com_load_after"] / df["com_capacity"]
df["stress_event_extreme"] = df["event_load_after"] / df["event_capacity"]

# ==============================
# SHEDDING LOGIC
# ==============================
df["shedding_normal"] = np.maximum(df["stress_res"] - 1, 0) * df["res_load"]
df["shedding_extreme"] = np.maximum(df["stress_res_extreme"] - 1, 0) * df["res_load_after"]

# ==============================
# SAVE EXTREME DATASET
# ==============================
df.to_csv(OUTPUT_FILE, index=False)

# ==============================
# 📊 VISUALIZATION
# ==============================

# 1️⃣ Stress Before vs After vs Extreme
plt.figure()
plt.plot(df["stress_res"].rolling(24).mean(), label="Before")
plt.plot(df["stress_res_after"].rolling(24).mean(), label="After Optimization")
plt.plot(df["stress_res_extreme"].rolling(24).mean(), label="Extreme Events")
plt.legend()
plt.title("Residential Stress Comparison")
plt.savefig(f"{FIGURE_DIR}/stress_comparison.png")
plt.close()

# 2️⃣ Load Shedding Comparison
plt.figure()
plt.plot(df["shedding_normal"].rolling(24).sum(), label="Normal Shedding")
plt.plot(df["shedding_extreme"].rolling(24).sum(), label="Extreme Shedding")
plt.legend()
plt.title("Load Shedding Under Normal vs Extreme")
plt.savefig(f"{FIGURE_DIR}/shedding_comparison.png")
plt.close()

# 3️⃣ Event Frequency
event_counts = df[[
    "event_heatwave",
    "event_festival",
    "event_ev_spike",
    "event_capacity_loss",
    "event_solar_drop"
]].sum()

event_counts.plot(kind="bar")
plt.title("Extreme Event Frequency")
plt.savefig(f"{FIGURE_DIR}/event_frequency.png")
plt.close()

print("✅ EXTREME EVENT SIMULATION + ANALYSIS COMPLETE")
print("📁 Extreme Dataset:", OUTPUT_FILE)
print("📊 Figures saved in:", FIGURE_DIR)
