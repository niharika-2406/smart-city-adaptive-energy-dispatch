"""
timegpt_forecast.py
Uses Nixtla TimeGPT via nixtla.NixtlaClient to forecast electricity load.

Requirements:
    pip install nixtla pandas matplotlib
Get API key from https://dashboard.nixtla.io
"""

import os
import argparse
import pandas as pd
import numpy as np
from nixtla import NixtlaClient
import matplotlib.pyplot as plt

# --------- CONFIG ----------
# API_KEY = os.getenv("NIXTLA_API_KEY")  # set NIXTLA_API_KEY in env, or pass directly below
# Or:
API_KEY = "nixak-cef8GXlbItbVmcEnJCPkT8lDYiZxFaoHfDWvUKp0lzqv4n5J5BxoaHmQq0OUCwDnhKr99GELiIS3HmId"

# If your dataset uses different names, adapt these:
TIME_COL = "datetime"
TARGET_COL = "total_load"

# --------- HELPERS ----------
def ensure_datetime_and_sort(df, time_col):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    return df

def build_future_exog(df, time_col, horizon, freq, exog_cols):
    """
    Create a future exogenous dataframe for the next `horizon` steps.
    Strategy used here (fast, generic):
      - For numeric exog: repeat the last observed value OR use simple seasonality.
      - For categorical flags (0/1): set to 0 except known events (user can override).
    Return dataframe containing rows for future timestamps and only exog_cols.
    """
    last_ts = df[time_col].max()
    future_index = pd.date_range(start=last_ts + pd.to_timedelta(1, unit="h"), periods=horizon, freq=freq)
    fut = pd.DataFrame({time_col: future_index})

    # For each exog column, create naive forecast: carry last value or use simple cycles for hour/day features
    for c in exog_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            # carry-forward the last value
            fut[c] = float(df[c].iloc[-1])
        else:
            # for non-numeric (if any), fallback to zeros
            fut[c] = 0

    return fut

# --------- MAIN ----------
def main(args):
    # 1) Load dataset
    df = pd.read_csv(args.data)
    df = ensure_datetime_and_sort(df, TIME_COL)

    # Choose freq and horizon (h in number of periods)
    # For hourly data set freq='H'
    freq = args.freq
    h = args.h

    # Basic validation
    if df.shape[0] < 10:
        raise SystemExit("Dataset too small for TimeGPT usage.")

    # 2) pick exogenous columns you want to pass to TimeGPT
    # Choose a subset of your features that are meaningful and available historically and (ideally) in future
    # Example: temperature, humidity, solar_irradiance, mobility_activity_index, ev_charging_activity,
    # is_festival, is_wedding_season, event_flag
    candidate_exogs = [
        "temperature","humidity","solar_irradiance","mobility_activity_index",
        "ev_charging_activity","is_festival","is_wedding_season","event_flag"
    ]
    exog_cols = [c for c in candidate_exogs if c in df.columns]
    print("Exogenous columns used:", exog_cols)

    # 3) instantiate Nixtla client
    api_key = API_KEY or args.api_key
    if not api_key:
        raise SystemExit("Provide NIXTLA_API_KEY environment variable or --api-key argument.")
    nixtla_client = NixtlaClient(api_key=api_key)

    # Optional: quick plot
    # nixtla_client.plot(df, time_col=TIME_COL, target_col=TARGET_COL)

    # 4) OPTIONAL: if you want to use multiple series or unique_id, adapt accordingly
    # TimeGPT single-series usage expects a dataframe with timestamp and value, plus exogenous columns (historical)
    input_df = df[[TIME_COL, TARGET_COL] + exog_cols].copy()

    # 5) prepare future exogenous dataframe for horizon `h`
    if exog_cols:
        future_exog = build_future_exog(df, TIME_COL, h, freq=args.freq, exog_cols=exog_cols)
    else:
        future_exog = None

    # 6) call TimeGPT forecast
    # You can select model variant: default None uses recommended model. Use model='timegpt-1-long-horizon' for long horizons.
    print("Requesting forecast from TimeGPT (h=%d, freq=%s) ..." % (h, freq))
    timegpt_fcst_df = nixtla_client.forecast(
        df=input_df,
        h=h,
        freq=freq,
        time_col=TIME_COL,
        target_col=TARGET_COL,
        X_df=future_exog,  # <-- Correct for your installed version
        level=[80, 90]
    )

    # 7) Save and plot results
    out_csv = args.output or "timegpt_forecast.csv"
    timegpt_fcst_df.to_csv(out_csv, index=False)
    print("Saved forecast to:", out_csv)

    # Quick plot: merge last historical window + forecast for visual continuity
    hist_plot_len = min(7*24, df.shape[0])  # last 7 days or less
    hist = df.tail(hist_plot_len)[[TIME_COL, TARGET_COL]].set_index(TIME_COL)
    fcst_plot = timegpt_fcst_df.set_index(TIME_COL)

    plt.figure(figsize=(12, 4))

    # Plot history
    plt.plot(hist.index, hist[TARGET_COL], label="History")

    # TimeGPT forecast mean
    plt.plot(fcst_plot.index, fcst_plot["TimeGPT"], label="Forecast")

    # Prediction interval (80%)
    plt.fill_between(
        fcst_plot.index,
        fcst_plot["TimeGPT-lo-80"],
        fcst_plot["TimeGPT-hi-80"],
        color="gray",
        alpha=0.25,
        label="80% PI"
    )

    plt.legend()
    plt.title("TimeGPT Forecast")
    plt.xlabel("time")
    plt.ylabel(TARGET_COL)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="SMART_CITY_FINAL_DATASET.csv")
    parser.add_argument("--api-key", type=str, default=None, help="TimeGPT API key (optional, or set NIXTLA_API_KEY)")
    parser.add_argument("--h", type=int, default=24, help="forecast horizon (periods)")
    parser.add_argument("--freq", type=str, default="H", help="pandas frequency string, e.g. 'H' for hourly")
    parser.add_argument("--output", type=str, default=None, help="forecast output CSV")
    args = parser.parse_args()
    main(args)
