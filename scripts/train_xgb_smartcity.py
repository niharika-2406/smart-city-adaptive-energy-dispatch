"""
train_xgb_smartcity.py

Train an XGBoost model on SMART_CITY_FINAL_DATASET.csv to predict 'total_load' (MW).
Outputs saved to artifacts_xgb/:
 - xgb_pipeline.joblib
 - xgb_metrics.json
 - xgb_test_predictions.csv
 - actual_vs_predicted.png
 - residuals_hist.png
 - feature_importance.png
Optional:
 - shap_summary.png

Run:
    python train_xgb_smartcity.py --data SMART_CITY_FINAL_DATASET.csv
"""

import os
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor, plot_importance

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# ---------------- CONFIG ----------------
RANDOM_SEED = 42
TARGET = "total_load"
DATE_COL = "datetime"
ARTIFACT_DIR = "artifacts_xgb"
N_ITER = 20
CV_SPLITS = 5
MAX_ONEHOT_CARD = 40   # only one-hot encode small-cardinality categoricals


# --------------- Helpers ----------------

def safe_mape(y_true, y_pred):
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# --------------- MAIN -------------------

def main(args):
    ensure_dir(ARTIFACT_DIR)
    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_SEED)

    # ========== LOAD DATA ==========
    print("Loading dataset:", args.data)
    df = pd.read_csv(args.data, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

    # Drop high-cardinality useless column
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    # Check target
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found!")

    # ========== TRAIN/TEST SPLIT ==========
    feature_cols = [c for c in df.columns if c not in [DATE_COL, TARGET]]

    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    cat_cols_filtered = [c for c in categorical_cols if df[c].nunique() < MAX_ONEHOT_CARD]

    print(f"Using {len(numeric_cols)} numeric and {len(cat_cols_filtered)} categorical (one-hot) columns.")

    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[numeric_cols + cat_cols_filtered]
    X_test  = test_df[numeric_cols + cat_cols_filtered]
    y_train = train_df[TARGET].astype(float)
    y_test  = test_df[TARGET].astype(float)
    dt_test = test_df[DATE_COL]

    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    # ========== PREPROCESSING PIPELINE ==========
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, numeric_cols),
        ("cat", cat_transformer, cat_cols_filtered)
    ], remainder="drop")

    # ========== MODEL ==========
    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_jobs=4,
        random_state=RANDOM_SEED,
        verbosity=0
    )

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", xgb)
    ])

    # ========== HYPERPARAMETER SEARCH ==========
    param_dist = {
        "model__n_estimators": [200, 400, 600],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [4, 6, 8],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__reg_alpha": [0, 0.1, 0.5],
        "model__reg_lambda": [1, 2, 5]
    }

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        verbose=2,
        random_state=RANDOM_SEED,
        n_jobs=1
    )

    print("Running hyperparameter search...")

    try:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        print("Best Params:", best_params)
    except Exception as e:
        print("Hyperparameter search FAILED:", e)
        print("Falling back to fixed hyperparameters...")
        fallback_params = {
            "model__n_estimators": 350,
            "model__learning_rate": 0.05,
            "model__max_depth": 6,
            "model__subsample": 0.9,
            "model__colsample_bytree": 0.9
        }
        pipeline.set_params(**fallback_params)
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        best_params = fallback_params

    # ========== SAVE MODEL ==========
    model_path = os.path.join(ARTIFACT_DIR, "xgb_pipeline.joblib")
    joblib.dump(best_model, model_path)
    print("Model saved at:", model_path)

    # ========== EVALUATION ==========
    y_pred = best_model.predict(X_test)

    # Fix for older sklearn: compute RMSE manually
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = safe_mape(y_test.values, y_pred)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_pct": mape,
        "best_params": best_params
    }

    with open(os.path.join(ARTIFACT_DIR, "xgb_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== MODEL METRICS ===")
    print(metrics)


    # Save predictions
    pred_df = pd.DataFrame({
        "datetime": dt_test,
        "actual": y_test,
        "predicted": y_pred
    })
    pred_df.to_csv(os.path.join(ARTIFACT_DIR, "xgb_test_predictions.csv"), index=False)

    # ========== PLOTS ==========
    # Actual vs Predicted (last 7 days)
    plt.figure(figsize=(12,4))
    plot_n = min(24*7, len(pred_df))
    plt.plot(pred_df["datetime"].iloc[-plot_n:], pred_df["actual"].iloc[-plot_n:], label="Actual")
    plt.plot(pred_df["datetime"].iloc[-plot_n:], pred_df["predicted"].iloc[-plot_n:], label="Predicted")
    plt.title("Actual vs Predicted (Last 7 Days)")
    plt.xlabel("Datetime")
    plt.ylabel("Total Load (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "actual_vs_predicted.png"))
    plt.close()

    # Residuals Histogram
    plt.figure(figsize=(6,4))
    plt.hist(pred_df["actual"] - pred_df["predicted"], bins=50)
    plt.title("Residual Distribution")
    plt.xlabel("Residual (MW)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "residuals_hist.png"))
    plt.close()

    # XGBoost Feature Importance
    model_inside = best_model.named_steps["model"]
    plt.figure(figsize=(8,6))
    plot_importance(model_inside, max_num_features=20, height=0.6)
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "feature_importance.png"))
    plt.close()

    # SHAP summary
    if SHAP_AVAILABLE:
        print("Generating SHAP summary...")
        sample_idx = np.random.choice(len(X_train), size=min(2000, len(X_train)), replace=False)
        X_sample = X_train.iloc[sample_idx]
        X_trans = best_model.named_steps["pre"].transform(X_sample)

        explainer = shap.Explainer(model_inside)
        shap_values = explainer(X_trans)
        shap.summary_plot(shap_values, X_trans, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, "shap_summary.png"))
        plt.close()
        print("SHAP saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="SMART_CITY_FINAL_DATASET.csv")
    args = parser.parse_args()
    main(args)
