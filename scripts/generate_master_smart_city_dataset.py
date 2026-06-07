import pandas as pd
import numpy as np
import os

# ======================================================
# CONFIG
# ======================================================
TIME_FILE = "time_calendar_2023_2025.csv"
SOCIAL_FILE = "smart_city_3year_social_domain.csv"
WEATHER_FILE = "real_hourly_weather_pune_2023_2025.csv"
SOLAR_FILE = "person3_combined_data.csv"  # optional, if exists
OUTPUT_FILE = "FINAL_SMART_CITY_DATASET_3YEAR.csv"

# ======================================================
# 1. TIME & CALENDAR DOMAIN
# ======================================================

def build_time_backbone():
    """Create a clean 3-year hourly time backbone."""
    start_date = "2023-01-01 00:00:00"
    end_date   = "2025-12-31 23:00:00"
    dt_index = pd.date_range(start=start_date, end=end_date, freq="H")
    df = pd.DataFrame({"datetime": dt_index})
    return df

def add_time_features(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear

    def get_season(m):
        if m in [3, 4, 5]:
            return "Summer"
        elif m in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "Winter"

    df["season"] = df["month"].apply(get_season)
    return df

print("🔹 Step 1: Building time backbone...")

if os.path.exists(TIME_FILE):
    time_df = pd.read_csv(TIME_FILE)
    if "datetime" not in time_df.columns:
        raise ValueError("time_calendar_2023_2025.csv must have a 'datetime' column.")
    time_df["datetime"] = pd.to_datetime(time_df["datetime"], dayfirst=True)
    time_df = time_df.sort_values("datetime").reset_index(drop=True)

    # Check if it is approx 3-year hourly
    expected_len = 3 * 365 * 24  # 26,280 (ignoring leap for simplicity)
    close_enough = abs(len(time_df) - expected_len) < 500

    if close_enough:
        df = time_df.copy()
        print(f"✅ Using TIME_FILE with {len(df)} rows.")
    else:
        print("⚠ TIME_FILE is not 3-year hourly. Regenerating clean backbone...")
        df = build_time_backbone()
else:
    print("⚠ TIME_FILE not found. Generating clean backbone...")
    df = build_time_backbone()

df = add_time_features(df)
print("✅ Time & calendar domain ready. Rows:", len(df))

# Ensure 'date' exists
df["date"] = df["datetime"].dt.date

# ======================================================
# 2. WEATHER & CLIMATE DOMAIN (REAL HOURLY WEATHER)
# ======================================================

print("\n🔹 Step 2: Weather & Climate Domain (Real Hourly Data)...")

if not os.path.exists(WEATHER_FILE):
    raise FileNotFoundError(
        "❌ Real hourly weather file not found: real_hourly_weather_pune_2023_2025.csv"
    )

# Load real hourly weather
wdf = pd.read_csv(WEATHER_FILE)
wdf["datetime"] = pd.to_datetime(wdf["datetime"])

# Expected columns in real weather file:
# temperature, humidity, wind_speed, cloud_cover, precipitation, pressure

required_weather_cols = [
    "datetime",
    "temperature",
    "humidity",
    "wind_speed",
    "cloud_cover",
    "precipitation",
    "pressure"
]

for col in required_weather_cols:
    if col not in wdf.columns:
        raise ValueError(f"❌ Missing required weather column in real file: {col}")

# Merge true hourly weather
df = df.merge(wdf, on="datetime", how="left")

# Feels-like temperature (derived)
df["feels_like_temp"] = (
    df["temperature"]
    + 0.1 * df["humidity"] / 10
    - 0.05 * df["wind_speed"]
)

# Safety fill (in case of rare missing hours)
df["temperature"] = df["temperature"].interpolate()
df["humidity"] = df["humidity"].interpolate()
df["wind_speed"] = df["wind_speed"].interpolate()
df["cloud_cover"] = df["cloud_cover"].interpolate()
df["precipitation"] = df["precipitation"].interpolate()
df["pressure"] = df["pressure"].interpolate()
df["feels_like_temp"] = df["feels_like_temp"].interpolate()

print("✅ Real hourly weather merged successfully.")


# ======================================================
# 3. SOCIAL EVENTS & HUMAN ACTIVITY DOMAIN
# ======================================================

print("\n🔹 Step 3: Social Events & Human Activity...")

if os.path.exists(SOCIAL_FILE):
    sdf = pd.read_csv(SOCIAL_FILE)
    if "datetime" in sdf.columns:
        sdf["datetime"] = pd.to_datetime(sdf["datetime"])
        df = df.merge(sdf, on="datetime", how="left")
        print("✅ Merged social domain from smart_city_3year_social_domain.csv")
    else:
        print("⚠ SOCIAL_FILE has no 'datetime'. Using synthetic social signals.")
else:
    print("⚠ SOCIAL_FILE not found. Using synthetic social signals.")

# Ensure all social columns exist
for col, default in [
    ("is_holiday", 0),
    ("is_festival", 0),
    ("festival_intensity", 0.0),
    ("is_wedding_season", 0),
    ("is_event_hour", 0),
    ("social_activity_index", 0.5),
]:
    if col not in df.columns:
        df[col] = default
    df[col] = df[col].fillna(default)

# If is_event_hour not properly set, define evenings
if "hour" not in df.columns:
    df["hour"] = df["datetime"].dt.hour

df.loc[(df["hour"].between(18, 23)) & (df["is_festival"] == 1), "is_event_hour"] = 1

# --- SAFETY: Recreate hour if merge altered it ---
if "hour" not in df.columns:
    df["hour"] = df["datetime"].dt.hour

print("✅ Social domain ready.")


# ================= SAFETY: FORCE SEASON COLUMN =================
if "season" not in df.columns:
    print("⚠ 'season' column missing. Recomputing from datetime...")
    
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["datetime"]).dt.month

    def _safety_season(m):
        if m in [3, 4, 5]:
            return "Summer"
        elif m in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "Winter"

    df["season"] = df["month"].apply(_safety_season)
# =================================================================

# ======================================================
# 4. SOLAR & RENEWABLE ENERGY DOMAIN
# ======================================================

print("\n🔹 Step 4: Solar & Renewable Domain...")

if os.path.exists(SOLAR_FILE):
    sol = pd.read_csv(SOLAR_FILE)
    if "datetime" in sol.columns:
        sol["datetime"] = pd.to_datetime(sol["datetime"])
        df = df.merge(sol, on="datetime", how="left")
        print("✅ Merged solar domain from person3_combined_data.csv")
    else:
        print("⚠ SOLAR_FILE has no 'datetime'. Using synthetic solar.")
else:
    print("⚠ SOLAR_FILE not found. Using synthetic solar.")

# solar_irradiance
if "solar_irradiance" not in df.columns:
    def synth_irr(row):
        # 0 at night, peak at noon
        if row["hour"] < 6 or row["hour"] > 18:
            return 0.0
        peak = 800 if row["season"] == "Summer" else 600 if row["season"] == "Monsoon" else 700
        x = (row["hour"] - 12) / 6.0
        return max(0.0, peak * np.exp(-x * x))
    df["solar_irradiance"] = df.apply(synth_irr, axis=1)

# solar_generation
if "solar_generation" not in df.columns:
    # Simple proportional model: MW ~ irradiance
    df["solar_generation"] = df["solar_irradiance"] * 0.02  # scale factor

# solar_penetration_ratio
if "day_of_year" not in df.columns:
    df["day_of_year"] = pd.to_datetime(df["datetime"]).dt.dayofyear

# solar_penetration_ratio
if "solar_penetration_ratio" not in df.columns:
    base_pen = 0.2 + 0.05 * np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["solar_penetration_ratio"] = np.clip(base_pen, 0.1, 0.35)
# ======================================================
# 5. MOBILITY & TRAFFIC ACTIVITY DOMAIN
# ======================================================

print("\n🔹 Step 5: Mobility & Traffic Domain...")

# is_peak_traffic_hour
df["is_peak_traffic_hour"] = 0
df.loc[(df["hour"].between(8, 10)) | (df["hour"].between(18, 21)), "is_peak_traffic_hour"] = 1

# traffic_density_index
traffic = np.where(df["is_peak_traffic_hour"] == 1,
                   np.random.uniform(0.7, 1.0, len(df)),
                   np.random.uniform(0.2, 0.6, len(df)))
df["traffic_density_index"] = traffic

# mobility_activity_index
mobility = 0.3 + 0.4 * (df["is_peak_traffic_hour"]) + 0.2 * df["is_event_hour"]
df["mobility_activity_index"] = np.clip(mobility, 0, 1)

# ev_charging_activity
ev = np.where(df["hour"].between(21, 23) | df["hour"].between(0, 2),
              np.random.uniform(0.5, 0.9, len(df)),
              np.random.uniform(0.1, 0.4, len(df)))
df["ev_charging_activity"] = ev

print("✅ Mobility & traffic domain ready.")

# --- SAFETY: Ensure is_weekend exists before load generation ---
if "is_weekend" not in df.columns:
    df["day_of_week"] = pd.to_datetime(df["datetime"]).dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# ======================================================
# 6. TOTAL LOAD (POWER GRID DOMAIN – PART 1)
# ======================================================

print("\n🔹 Step 6: Generating Total City Load (MW)...")

def gen_total_load(row):
    base = 120  # base MW
    temp_factor = 1 + 0.035 * (row["temperature"] - 25)
    hour_factor = 1.25 if 18 <= row["hour"] <= 22 else 1.0
    festival_factor = 1.3 if row["is_festival"] == 1 else 1.0
    weekend_factor = 1.1 if row["is_weekend"] == 1 else 1.0
    social_factor = 1.0 + 0.2 * row["social_activity_index"]
    noise = np.random.normal(0, 4)
    val = base * temp_factor * hour_factor * festival_factor * weekend_factor * social_factor + noise
    return max(val, 60)

df["total_load"] = df.apply(gen_total_load, axis=1)

# net_load after solar offset
df["net_load"] = df["total_load"] - df["solar_generation"]

print("✅ Total & net load ready.")

# ======================================================
# 7. URBAN ZONING & LAND-USE DOMAIN
# ======================================================

print("\n🔹 Step 7: Zoning & Load Distribution...")

def zone_weights(row):
    h = row["hour"]
    weekend = row["is_weekend"]
    festival = row["is_festival"]

    # Festival / wedding intense behaviour
    if festival == 1:
        if h < 10:
            return 0.50, 0.25, 0.25
        elif h < 17:
            return 0.40, 0.30, 0.30
        elif h < 23:
            return 0.35, 0.20, 0.45
        else:
            return 0.50, 0.20, 0.30
    # Weekend
    elif weekend == 1:
        if h < 8:
            return 0.70, 0.15, 0.15
        elif h < 14:
            return 0.50, 0.30, 0.20
        elif h < 18:
            return 0.45, 0.25, 0.30
        elif h < 23:
            return 0.40, 0.20, 0.40
        else:
            return 0.55, 0.20, 0.25
    # Normal weekday
    else:
        if h < 6:
            return 0.65, 0.20, 0.15
        elif h < 10:
            return 0.45, 0.45, 0.10
        elif h < 17:
            return 0.35, 0.55, 0.10
        elif h < 22:
            return 0.50, 0.30, 0.20
        else:
            return 0.60, 0.20, 0.20

df[["residential_weight", "commercial_weight", "event_weight"]] = \
    df.apply(lambda r: pd.Series(zone_weights(r)), axis=1)

# industrial_weight (optional, keep 0 for now)
df["industrial_weight"] = 0.0

# zone_activity_index = max of zone weights
df["zone_activity_index"] = df[["residential_weight", "commercial_weight", "event_weight"]].max(axis=1)

# ZONE LOADS
df["res_load"] = df["total_load"] * df["residential_weight"]
df["com_load"] = df["total_load"] * df["commercial_weight"]
df["event_load"] = df["total_load"] * df["event_weight"]

print("✅ Zoning & derived zone loads ready.")

# ======================================================
# 8. GRID CAPACITY, FLEXIBILITY & STRESS
# ======================================================

print("\n🔹 Step 8: Grid Capacity, Flexibility & Stress...")

TOTAL_PEAK = df["total_load"].max()

df["res_capacity"] = 0.45 * TOTAL_PEAK
df["com_capacity"] = 0.35 * TOTAL_PEAK
df["event_capacity"] = 0.25 * TOTAL_PEAK

# Per-zone flexibility
df["flex_res"] = 0.25
df["flex_com"] = 0.15
df["flex_event"] = 0.08

# Overall flexible_load_ratio (for your schema)
df["flexible_load_ratio"] = (
    df["residential_weight"] * df["flex_res"] +
    df["commercial_weight"] * df["flex_com"] +
    df["event_weight"] * df["flex_event"]
)

df["storage_available"] = 0.08 * TOTAL_PEAK  # MWh-equivalent heuristic

# Stress before compensation
df["stress_res"] = df["res_load"] / df["res_capacity"]
df["stress_com"] = df["com_load"] / df["com_capacity"]
df["stress_event"] = df["event_load"] / df["event_capacity"]

print("✅ Stress (before compensation) ready.")

# ======================================================
# 9. COMPENSATION LOGIC
# ======================================================

print("\n🔹 Step 9: Applying Compensation Logic...")

def compensate(row):
    res, com, ev = row["res_load"], row["com_load"], row["event_load"]
    sr, sc, se = row["stress_res"], row["stress_com"], row["stress_event"]
    h = row["hour"]

    flex_res = row["flex_res"] * res
    flex_com = row["flex_com"] * com

    # Rule 1: Event overload
    if se > 1.0:
        shift = 0.10 * flex_com
        ev += shift
        com -= shift

    # Rule 2: Residential evening overload
    if sr > 1.0 and 19 <= h <= 23:
        shift = 0.15 * flex_res
        res -= shift

    # Rule 3: multi-zone stress, sacrifice commercial
    overloads = sum([sr > 1.0, sc > 1.0, se > 1.0])
    if overloads >= 2:
        com *= 0.95

    return res, com, ev

df[["res_load_after", "com_load_after", "event_load_after"]] = \
    df.apply(lambda r: pd.Series(compensate(r)), axis=1)

df["stress_res_after"] = df["res_load_after"] / df["res_capacity"]
df["stress_com_after"] = df["com_load_after"] / df["com_capacity"]
df["stress_event_after"] = df["event_load_after"] / df["event_capacity"]

print("✅ Compensation applied and post-stress computed.")

# ======================================================
# 10. FINAL COLUMN SANITY (ensure all specified columns exist)
# ======================================================

print("\n🔹 Step 10: Final column sanity check...")

required_cols = [
    "datetime", "hour", "day_of_week", "is_weekend", "month", "season", "day_of_year",
    "temperature", "feels_like_temp", "humidity", "wind_speed", "cloud_cover", "precipitation", "pressure",
    "is_holiday", "is_festival", "festival_intensity", "is_wedding_season", "is_event_hour", "social_activity_index",
    "solar_irradiance", "solar_generation", "solar_penetration_ratio", "net_load",
    "residential_weight", "commercial_weight", "event_weight", "industrial_weight", "zone_activity_index",
    "is_peak_traffic_hour", "traffic_density_index", "mobility_activity_index", "ev_charging_activity",
    "total_load", "res_capacity", "com_capacity", "event_capacity", "flexible_load_ratio", "storage_available",
    "stress_res", "stress_com", "stress_event"
]

for col in required_cols:
    if col not in df.columns:
        # If somehow missing, set a safe default
        df[col] = 0

print("✅ All required columns are present.")

# ======================================================
# 11. SAVE FINAL DATASET
# ======================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\n🎉 FINAL MASTER DATASET GENERATED SUCCESSFULLY!")
print(f"📁 File: {OUTPUT_FILE}")
print("📊 Rows:", len(df))
print("📌 Columns:", len(df.columns))
