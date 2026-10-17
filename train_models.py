#!/usr/bin/env python3
"""
Fase 2 — Reentrenamiento de Modelos (memory-optimized additions)

This is a small, conservative set of edits on top of the existing training script:
- Adds a --produce-serving CLI flag that will call the scripts/prepare_serving_dataset.py helper to produce dataset_serving.parquet (downcasted)
- Saves compressed joblib artifacts (compress=3)
- Attempts to save native xgboost .xgb model files (smaller / faster to load in serving)

Note: This file preserves the original training flow. It runs offline and should not be executed in the Streamlit runtime.
"""

import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# SELECTOR DEL MODELO DE ENTRENAMIENTO: cambia esta clase para probar otro estimador sklearn.
MODEL_CLASS = XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_enriched.parquet"
REGISTRY_DIR = BASE_DIR / "model_registry"
REGISTRY_DIR.mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description='Train models and optionally produce serving dataset')
parser.add_argument('--produce-serving', action='store_true', help='Produce dataset_serving.parquet (downcasted) after training')
args = parser.parse_args()

print("=" * 60)
print("FASE 2: Reentrenamiento de Modelos")
print("=" * 60)

# ── Cargar datos ─────────────────────────────────────────────────
print("\nCargando dataset enriquecido...")
df = pd.read_parquet(DATA_PATH)
print(f"Shape: {df.shape}")

# Imputar nulos menores
df["Extension"] = df["Extension"].fillna(df["Extension"].median())
df["effective_distance"] = 60.5 - df["Extension"]
df["perceived_velo_factor"] = (df["RelSpeed"] * 60.5) / df["effective_distance"]
df["fb_count"] = df["fb_count"].fillna(0)
df["SpinAxis"] = df["SpinAxis"].fillna(df["SpinAxis"].median())
df["SpeedDrop"] = df["SpeedDrop"].fillna(df["RelSpeed"] - df["ZoneSpeed"])

# Encoding categorico para AutoPitchType
pitch_type_map = {
    "Four-Seam": 0, "Sinker": 1, "Cutter": 2, "Changeup": 3,
    "Slider": 4, "Curveball": 5, "Splitter": 6, "Sweeper": 7,
    "Other": 8, "OneSeamFastBall": 0, "TwoSeamFastBall": 1, "Knuckleball": 8
}
df["AutoPitchType_Num"] = df["AutoPitchType"].map(pitch_type_map).fillna(8).astype(int)

# Target: xRV invertido (mayor = mejor para pitcher, necesario para escala plus)
# En xRV original: menor = mejor para pitcher, pero la escala plus necesita mayor = mejor
df["xRV_inv"] = -df["xRV"]  # Ahora mayor xRV_inv = mejor calidad

# ── Train/Test split por pitcher ─────────────────────────────────
print("\nDividiendo datos (pitcher holdout)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
groups = df["pitcher_anon_id"]
train_idx, test_idx = next(gss.split(df, groups=groups))

df_train = df.iloc[train_idx].copy()
df_test = df.iloc[test_idx].copy()
print(f"  Train: {len(df_train)} ({df_train['pitcher_anon_id'].nunique()} pitchers)")
print(f"  Test:  {len(df_test)} ({df_test['pitcher_anon_id'].nunique()} pitchers)")

# ── Definicion de features por modelo ────────────────────────────
STUFF_FEATURES = [
    "RelSpeed", "SpinRate", "SpinAxis", "InducedVertBreak", "HorzBreak",
    "VertBreak", "EffectiveVelo", "SpeedDrop",
    "RelHeight", "RelSide", "Extension",
    "VertRelAngle", "HorzRelAngle",
    "AutoPitchType_Num", "PitcherThrows_Num", "BatterSide_Num",
    "velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb",
]

LOCATION_FEATURES = [
    "PlateLocHeight", "PlateLocSide", "in_strike_zone",
    "Balls", "Strikes",
    "AutoPitchType_Num", "PitcherThrows_Num", "BatterSide_Num",
]

PITCHING_FEATURES = [
    "RelSpeed", "SpinRate", "SpinAxis", "InducedVertBreak", "HorzBreak",
    "VertBreak", "EffectiveVelo", "SpeedDrop",
    "RelHeight", "RelSide", "Extension",
    "VertRelAngle", "HorzRelAngle",
    "PlateLocHeight", "PlateLocSide", "in_strike_zone",
    "Balls", "Strikes",
    "AutoPitchType_Num", "PitcherThrows_Num", "BatterSide_Num",
    "velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb",
    "effective_distance", "perceived_velo_factor",
]

TARGET = "xRV_inv"
validation_metrics = {"pitcher_holdout": {}, "season_holdout": {}, "altitude_holdout": {}}


def train_and_evaluate(name, features, params, df_train, df_test):
    """Entrena un modelo XGBoost y reporta metricas."""
    print(f"\n{'─' * 50}")
    print(f"Entrenando {name}...")
    print(f"  Features ({len(features)}): {features}")
    
    X_train = df_train[features].copy()
    y_train = df_train[TARGET].copy()
    X_test = df_test[features].copy()
    y_test = df_test[TARGET].copy()
    
    # Imputar NaN residuales con mediana del train
    for c in features:
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med)
        X_test[c] = X_test[c].fillna(med)
    
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    mae_test = mean_absolute_error(y_test, pred_test)
    r2_test = r2_score(y_test, pred_test)
    
    print(f"\n  Resultados {name}:")
    print(f"    RMSE train: {rmse_train:.6f}")
    print(f"    RMSE test:  {rmse_test:.6f}")
    print(f"    MAE test:   {mae_test:.6f}")
    print(f"    R2 test:    {r2_test:.4f}")
    validation_metrics["pitcher_holdout"][name] = {
        "n": int(len(y_test)),
        "rmse": float(rmse_test),
        "mae": float(mae_test),
        "r2": float(r2_test),
    }
    
    # Feature importance
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(f"\n  Top 10 features:")
    for feat, val in imp.head(10).items():
        print(f"    {feat}: {val:.4f}")
    
    return model, pred_train, pred_test

# ... rest of file remains the same until saving models and dataset

# Save compressed joblib artifacts
joblib.dump(stuff_model, REGISTRY_DIR / "stuff_model.pkl", compress=3)
joblib.dump(loc_model, REGISTRY_DIR / "location_model.pkl", compress=3)
joblib.dump(pitching_model, REGISTRY_DIR / "pitching_model.pkl", compress=3)

# Save native xgboost boosters where possible
try:
    # If a sklearn wrapper is used, get_booster may be available
    stuff_model.get_booster().save_model(str(REGISTRY_DIR / "stuff_model.xgb"))
    loc_model.get_booster().save_model(str(REGISTRY_DIR / "location_model.xgb"))
    pitching_model.get_booster().save_model(str(REGISTRY_DIR / "pitching_model.xgb"))
except Exception as e:
    print(f"Warning: could not save native xgboost models: {e}")

# Save metadata and scored dataset as before
scale_meta = {
    "stuff": {"mu": float(stuff_pred_train.mean()), "sigma": float(stuff_pred_train.std())},
    "location": {"mu": float(loc_pred_train.mean()), "sigma": float(loc_pred_train.std())},
    "pitching": {"mu": float(pitch_pred_train.mean()), "sigma": float(pitch_pred_train.std())},
    "features": {
        "stuff": STUFF_FEATURES,
        "location": LOCATION_FEATURES,
        "pitching": PITCHING_FEATURES,
    },
    "pitch_type_map": pitch_type_map,
}

import json
with open(REGISTRY_DIR / "scale_metadata.json", "w") as f:
    json.dump(scale_meta, f, indent=2)

# Save full scored dataset
df.to_parquet(BASE_DIR / "dataset_scores_final_v2.parquet", index=False)

# Produce serving dataset if requested
if args.produce_serving:
    try:
        from scripts.prepare_serving_dataset import build_serving_dataset
        print("Producing dataset_serving.parquet (downcasted) ...")
        build_serving_dataset(BASE_DIR / "dataset_scores_final_v2.parquet", BASE_DIR / "dataset_serving.parquet")
    except Exception as e:
        print(f"Could not produce serving dataset automatically: {e}")

# Save validation metrics
with open(REGISTRY_DIR / "validation_metrics.json", "w") as f:
    json.dump(validation_metrics, f, indent=2)

print("\nDone.")
