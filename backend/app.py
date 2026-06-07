from flask import Flask, jsonify, render_template, request
import pandas as pd
import os

from flask import Flask, jsonify, render_template, request
import pandas as pd
import os

# ✅ ADD THIS HERE (BEFORE app = Flask(...))
ZONE_FACTORS = {
    "all": 1.0,

    # West Pune – IT Hubs
    "hinjewadi": 0.16,
    "baner": 0.08,
    "wakad": 0.07,
    "aundh": 0.05,

    # East Pune – IT Parks
    "kharadi": 0.09,
    "hadapsar": 0.07,
    "magarpatta": 0.05,
    "viman_nagar": 0.03,

    # North Pune – Industrial
    "pimpri": 0.09,
    "chinchwad": 0.06,
    "bhosari": 0.05,

    # Central & South
    "shivajinagar": 0.06,
    "camp": 0.04,
    "kondhwa": 0.05,
    "katraj": 0.05
}

app = Flask(__name__, static_folder="static", template_folder="templates")

app = Flask(__name__, static_folder="static", template_folder="templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
ADAPTIVE_FILE = os.path.join(DATA_DIR, "processed", "SMART_CITY_ADAPTIVE_DISPATCH.csv")
EXTREME_FILE = os.path.join(DATA_DIR, "processed", "SMART_CITY_EXTREME_EVENTS.csv")
TIMEGPT_FILE = os.path.join(DATA_DIR, "forecasts", "timegpt_forecast.csv")
XGB_FILE = os.path.join(DATA_DIR, "forecasts", "xgb_test_predictions.csv")

df_timegpt = pd.read_csv(TIMEGPT_FILE)
df_xgb = pd.read_csv(XGB_FILE)

df_timegpt.fillna(0, inplace=True)
df_xgb.fillna(0, inplace=True)

print("✅ TimeGPT Forecast Loaded:", df_timegpt.shape)
print("✅ XGB Forecast Loaded:", df_xgb.shape)
print("✅ TimeGPT Columns:", df_timegpt.columns)
print("✅ XGB Columns:", df_xgb.columns)



df_adaptive = pd.read_csv(ADAPTIVE_FILE)
df_extreme = pd.read_csv(EXTREME_FILE)
df_adaptive.fillna(0, inplace=True)
df_extreme.fillna(0, inplace=True)


print(df_adaptive.tail(1)[[
    "total_load",
    "net_load",
    "adaptive_net_load",
    "risk_index"
]])

print("✅ Data Loaded:")
print("Adaptive:", df_adaptive.shape)
print("Extreme:", df_extreme.shape)

@app.route("/")
def home():
    return render_template("dashboard.html")

import math

@app.route("/api/status")
def status():
    zone = request.args.get("zone", "all").lower()

    # ✅ ADD THIS LINE (virtual zone factor)
    factor = ZONE_FACTORS.get(zone, 1.0)

    filtered = df_adaptive.copy()

    if zone != "all" and "zone" in filtered.columns:
        filtered["zone"] = filtered["zone"].astype(str).str.lower()
        filtered = filtered[filtered["zone"].str.contains(zone)]

    if filtered.empty:
        return jsonify({
            "stress": 0,
            "storage": 0,
            "shedding": 0,
            "risk": 0,
            "zonal_load": 0   # ✅ keep frontend safe
        })

    latest = filtered.iloc[-1]

    total_load = float(latest.get("total_load", 0))
    net_load = float(latest.get("net_load", 0))

    effective_load = net_load if net_load > 0 else total_load

    # ✅ APPLY ZONE SCALING HERE (this is the only real change)
    zonal_load = effective_load * factor

    storage = round(zonal_load * 0.05, 2)
    shedding = round(zonal_load * 0.02, 2)
    stress = round(float(latest.get("risk_index", 0)) * 100, 2)
    risk = float(latest.get("risk_index", 0))

    return jsonify({
        "stress": stress,
        "storage": storage,
        "shedding": shedding,
        "risk": risk,
        "zonal_load": round(zonal_load, 2)   # ✅ added field
    })


@app.route("/api/forecast")
def forecast():
    try:
        model = request.args.get("model", "timegpt").lower()
        zone = request.args.get("zone", "all").lower()

        factor = ZONE_FACTORS.get(zone, 1.0)

        # ✅ Select correct dataframe
        if model == "timegpt":
            df = df_timegpt.copy()
        elif model == "xgb":
            df = df_xgb.copy()
        else:
            df = df_timegpt.copy()

        # ✅ AUTO-DETECT FIRST NUMERIC COLUMN
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_cols:
            return jsonify({"error": "No numeric forecast column found"}), 500

        col = numeric_cols[0]   # ✅ Correct numeric prediction column

        data = df[[col]].tail(24).copy()

        # ✅ Apply zone scaling
        data[col] = data[col] * float(factor)

        data.columns = ["load"]

        return jsonify(data.to_dict(orient="records"))

    except Exception as e:
        print("❌ FORECAST API ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/api/adaptive")
def adaptive():
    zone = request.args.get("zone", "all").lower()

    # ✅ Virtual zone scaling
    factor = ZONE_FACTORS.get(zone, 1.0)

    filtered = df_adaptive.copy()

    # ✅ Keep your original zone filtering logic (if ever needed)
    if zone != "all" and "zone" in filtered.columns:
        filtered["zone"] = filtered["zone"].astype(str).str.lower()
        filtered = filtered[filtered["zone"].str.contains(zone)]

    if filtered.empty:
        return jsonify([])

    data = filtered.tail(300).copy()

    # ✅ APPLY ZONE SCALING (THIS IS THE MAIN CHANGE)
    if "adaptive_net_load" in data.columns:
        data["adaptive_net_load"] = data["adaptive_net_load"] * factor
    if "net_load" in data.columns:
        data["net_load"] = data["net_load"] * factor
    if "total_load" in data.columns:
        data["total_load"] = data["total_load"] * factor

    return jsonify(data.to_dict(orient="records"))




@app.route("/api/extreme")
def extreme():
    return jsonify(
        df_extreme.tail(200).fillna(0).to_dict(orient="records")
    )

if __name__ == "__main__":
    app.run(debug=True)
