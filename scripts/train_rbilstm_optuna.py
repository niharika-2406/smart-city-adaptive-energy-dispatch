"""
Lightweight RBiLSTM + Attention + Optuna (FULLY PATCHED)
- Works on CPU
- Handles categorical features
- Saves model, scaler, encoders, metrics
- NO 'squared=False' crash
"""

import argparse
import os
import numpy as np
import pandas as pd
import json
import joblib
import optuna
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

TARGET = "total_load"
DATE_COL = "datetime"
ARTIFACT_DIR = "artifacts_rbilstm_light"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ============================================================
# ATTENTION LAYER
# ============================================================
class Attention(layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform"
        )
        self.b = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros"
        )
        self.u = self.add_weight(
            shape=(input_shape[-1],),
            initializer="glorot_uniform"
        )

    def call(self, x):
        v = tf.tanh(tf.matmul(x, self.W) + self.b)
        vu = tf.tensordot(v, self.u, axes=1)
        alphas = tf.nn.softmax(vu)
        return tf.reduce_sum(x * tf.expand_dims(alphas, -1), axis=1)


# ============================================================
# LIGHT RBiLSTM MODEL BUILDER
# ============================================================
def build_light_model(input_shape, units, dropout, lr):
    inp = layers.Input(shape=input_shape)

    x = layers.Bidirectional(
        layers.LSTM(units, return_sequences=True, dropout=dropout)
    )(inp)

    att = Attention()(x)

    d = layers.Dense(32, activation="relu")(att)
    d = layers.Dropout(dropout)(d)
    out = layers.Dense(1)(d)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ============================================================
# SEQUENCE GENERATOR
# ============================================================
def create_sequences(data, target, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)


# ============================================================
# OPTUNA OBJECTIVE FUNCTION (PATCHED)
# ============================================================
def objective(trial, X_train, y_train, X_val, y_val, n_features):

    units = trial.suggest_int("units", 16, 96)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    lookback = trial.suggest_int("lookback", 24, 72)

    X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, lookback)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, lookback)

    model = build_light_model(
        input_shape=(lookback, n_features),
        units=units,
        dropout=dropout,
        lr=lr
    )

    es = EarlyStopping(patience=3, restore_best_weights=True)

    model.fit(
        X_tr_seq, y_tr_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=12,
        batch_size=batch_size,
        verbose=1,
        callbacks=[es]
    )

    pred = model.predict(X_val_seq, verbose=0)

    # PATCH FIX → RMSE without squared argument
    rmse = np.sqrt(mean_squared_error(y_val_seq, pred))

    return rmse


# ============================================================
# MAIN SCRIPT
# ============================================================
def main(args):

    print("\n📥 Loading dataset…\n")
    df = pd.read_csv(args.data, parse_dates=[DATE_COL]).sort_values(DATE_COL)

    # Detect object columns
    print("🔎 Detecting categorical columns…")
    categorical_cols = [
        col for col in df.columns
        if df[col].dtype == 'object' and col not in [DATE_COL, TARGET]
    ]

    print("Categorical columns:", categorical_cols)

    label_encoders = {}

    # Encode all categoricals
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    joblib.dump(label_encoders, os.path.join(ARTIFACT_DIR, "label_encoders.joblib"))
    print("✅ Label encoders saved.\n")

    feature_cols = [c for c in df.columns if c not in [DATE_COL, TARGET]]
    X = df[feature_cols].values
    y = df[TARGET].values

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    print("✅ Scaler saved.\n")

    # Split
    N = len(df)
    train_end = int(0.7 * N)
    val_end = int(0.85 * N)

    X_train, y_train = X_scaled[:train_end], y[:train_end]
    X_val, y_val = X_scaled[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X_scaled[val_end:], y[val_end:]

    n_features = X_train.shape[1]

    print("\n🔥 Starting Optuna hyperparameter tuning…\n")

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, n_features),
        n_trials=args.trials,
        show_progress_bar=True
    )

    print("\n🏆 Best parameters:", study.best_params)
    best = study.best_params
    lookback = best["lookback"]

    # Final training set (train + val)
    X_trainval = X_scaled[:val_end]
    y_trainval = y[:val_end]

    X_trainval_seq, y_trainval_seq = create_sequences(X_trainval, y_trainval, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, lookback)

    final_model = build_light_model(
        (lookback, n_features),
        units=best["units"],
        dropout=best["dropout"],
        lr=best["lr"]
    )

    print("\n⚙ Training final model…\n")
    final_model.fit(
        X_trainval_seq, y_trainval_seq,
        epochs=18,
        batch_size=best["batch_size"],
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=1
    )

    # Save model
    model_path = os.path.join(ARTIFACT_DIR, "rbilstm_light.h5")
    final_model.save(model_path)
    print("\n✅ Model saved:", model_path)

    # Evaluate
    pred = final_model.predict(X_test_seq).ravel()

    # PATCH FIX → RMSE without squared=False
    rmse = np.sqrt(mean_squared_error(y_test_seq, pred))
    mae = mean_absolute_error(y_test_seq, pred)
    r2 = r2_score(y_test_seq, pred)

    print("\n📊 FINAL METRICS")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")

    json.dump(
        {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)},
        open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w"),
        indent=2
    )
    print("📁 Metrics saved.\n")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="SMART_CITY_FINAL_DATASET.csv")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    main(args)
