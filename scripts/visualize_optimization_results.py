import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# CONFIG
# ======================================================
INPUT_CSV = "data/proccessed/SMART_CITY_OPTIMIZED_DISPATCH.csv"
OUTPUT_DIR = "reports/opt_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# LOAD DATA
# ======================================================
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Optimized dataset not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
print("[INFO] Loaded:", INPUT_CSV)

# ======================================================
# 1️⃣ STRESS BEFORE vs AFTER OPTIMIZATION
# ======================================================

print("[INFO] Plotting stress before vs after...")

plt.figure()
plt.plot(df["stress_res"], label="Residential BEFORE", alpha=0.5)
plt.plot(df["stress_res_after"], label="Residential AFTER", linewidth=2)
plt.legend()
plt.title("Residential Stress Before vs After Optimization")
plt.xlabel("Time Index")
plt.ylabel("Stress Ratio")
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/stress_res_before_after.png", dpi=150)
plt.close()

plt.figure()
plt.plot(df["stress_com"], label="Commercial BEFORE", alpha=0.5)
plt.plot(df["stress_com_after"], label="Commercial AFTER", linewidth=2)
plt.legend()
plt.title("Commercial Stress Before vs After Optimization")
plt.xlabel("Time Index")
plt.ylabel("Stress Ratio")
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/stress_com_before_after.png", dpi=150)
plt.close()

plt.figure()
plt.plot(df["stress_event"], label="Event BEFORE", alpha=0.5)
plt.plot(df["stress_event_after"], label="Event AFTER", linewidth=2)
plt.legend()
plt.title("Event Stress Before vs After Optimization")
plt.xlabel("Time Index")
plt.ylabel("Stress Ratio")
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/stress_event_before_after.png", dpi=150)
plt.close()

print("✅ Stress comparison plots saved.")

# ======================================================
# 2️⃣ SHEDDING: NORMAL vs EXTREME EVENT SIMULATION
# ======================================================

print("[INFO] Creating extreme-event load scenario...")

# ---- Create a synthetic "extreme" version of demand ----
df_extreme = df.copy()

# Extreme multipliers
heatwave_boost = 1.20      # +20% demand
festival_boost = 1.15     # +15% demand
ev_spike_boost = 1.10     # +10% night load

df_extreme["total_load_extreme"] = df["total_load"] * heatwave_boost

# Extra night EV spike
night_mask = (df_extreme["datetime"].astype(str).str.contains(":2")) if "datetime" in df_extreme else np.zeros(len(df_extreme))
df_extreme["total_load_extreme"] *= np.where(df_extreme["hour"].between(20, 23), ev_spike_boost, 1)

# Recalculate extreme shedding approximately
df_extreme["shed_extreme"] = np.clip(
    df_extreme["total_load_extreme"] - df_extreme["total_load"], 0, None
)

# ======================================================
# 3️⃣ SHEDDING COMPARISON PLOT
# ======================================================

print("[INFO] Plotting shedding comparison...")

plt.figure()
plt.plot(df["total_shed_after"], label="Shedding (Normal)", alpha=0.7)
plt.plot(df_extreme["shed_extreme"], label="Shedding (Extreme)", linewidth=2)
plt.legend()
plt.title("Shedding: Normal vs Extreme Event Scenario")
plt.xlabel("Time Index")
plt.ylabel("Shedding (MW)")
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/shedding_normal_vs_extreme.png", dpi=150)
plt.close()

print("✅ Shedding comparison plot saved.")

# ======================================================
# 4️⃣ SUMMARY METRICS REPORT
# ======================================================

summary = {
    "avg_shedding_normal": float(df["total_shed_after"].mean()),
    "max_shedding_normal": float(df["total_shed_after"].max()),
    "avg_shedding_extreme": float(df_extreme["shed_extreme"].mean()),
    "max_shedding_extreme": float(df_extreme["shed_extreme"].max()),
    "overload_hours_normal": int(df["grid_overload_flag"].sum())
}

summary_df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
summary_df.to_csv(f"{OUTPUT_DIR}/optimization_summary.csv", index=False)

print("✅ Optimization summary saved.")

print("\n🎉 VISUALIZATION COMPLETE")
print("All plots & report saved in → reports/figures/")
