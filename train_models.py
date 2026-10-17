"""
Fase 2 — Reentrenamiento de Modelos
=====================================
Entrena Stuff+, Location+ y Pitching+ con features correctas,
escala Plus, validacion holdout y guarda modelos nuevos.
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

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
    
    model = MODEL_CLASS(**params)
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


def compute_plus_scale(predictions, ref_predictions):
    """Convierte predicciones crudas a escala Plus (media=100, std=10)."""
    mu = ref_predictions.mean()
    sigma = ref_predictions.std()
    if sigma < 1e-10:
        sigma = 1.0
    z = (predictions - mu) / sigma
    return 100 + 10 * z


# ══════════════════════════════════════════════════════════════════
# MODELO 1: STUFF+
# ══════════════════════════════════════════════════════════════════
stuff_params = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "enable_categorical": False,
    "early_stopping_rounds": 30,
}

stuff_model, stuff_pred_train, stuff_pred_test = train_and_evaluate(
    "Stuff+", STUFF_FEATURES, stuff_params, df_train, df_test
)

# ══════════════════════════════════════════════════════════════════
# MODELO 2: LOCATION+
# ══════════════════════════════════════════════════════════════════
loc_params = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "enable_categorical": False,
    "early_stopping_rounds": 30,
}

loc_model, loc_pred_train, loc_pred_test = train_and_evaluate(
    "Location+", LOCATION_FEATURES, loc_params, df_train, df_test
)

# ══════════════════════════════════════════════════════════════════
# MODELO 3: PITCHING+
# ══════════════════════════════════════════════════════════════════
pitch_params = {
    "objective": "reg:squarederror",
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "enable_categorical": False,
    "early_stopping_rounds": 30,
}

pitching_model, pitch_pred_train, pitch_pred_test = train_and_evaluate(
    "Pitching+", PITCHING_FEATURES, pitch_params, df_train, df_test
)

# ══════════════════════════════════════════════════════════════════
# ESCALA PLUS
# ══════════════════════════════════════════════════════════════════
print(f"\n{'─' * 50}")
print("Calculando escala Plus...")

# Usar predicciones del train como referencia para la escala
stuff_plus_train = compute_plus_scale(stuff_pred_train, stuff_pred_train)
stuff_plus_test = compute_plus_scale(stuff_pred_test, stuff_pred_train)

loc_plus_train = compute_plus_scale(loc_pred_train, loc_pred_train)
loc_plus_test = compute_plus_scale(loc_pred_test, loc_pred_train)

pitch_plus_train = compute_plus_scale(pitch_pred_train, pitch_pred_train)
pitch_plus_test = compute_plus_scale(pitch_pred_test, pitch_pred_train)

for name, train_s, test_s in [
    ("Stuff+", stuff_plus_train, stuff_plus_test),
    ("Location+", loc_plus_train, loc_plus_test),
    ("Pitching+", pitch_plus_train, pitch_plus_test),
]:
    print(f"\n  {name} escala Plus:")
    print(f"    Train — media: {train_s.mean():.1f}, std: {train_s.std():.1f}, rango: [{train_s.min():.1f}, {train_s.max():.1f}]")
    print(f"    Test  — media: {test_s.mean():.1f}, std: {test_s.std():.1f}, rango: [{test_s.min():.1f}, {test_s.max():.1f}]")

# ══════════════════════════════════════════════════════════════════
# GUARDAR MODELOS Y METADATA
# ══════════════════════════════════════════════════════════════════
print(f"\n{'─' * 50}")
print("Guardando modelos...")

# Guardar modelos
joblib.dump(stuff_model, REGISTRY_DIR / "stuff_model.pkl")
joblib.dump(loc_model, REGISTRY_DIR / "location_model.pkl")
joblib.dump(pitching_model, REGISTRY_DIR / "pitching_model.pkl")

# Guardar metadata de escala (mu y sigma de referencia)
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

# Guardar dataset con scores recalculados
print("Recalculando scores en dataset completo...")
X_all_stuff = df[STUFF_FEATURES].copy()
X_all_loc = df[LOCATION_FEATURES].copy()
X_all_pitch = df[PITCHING_FEATURES].copy()

# Imputar NaN residuales con mediana
for feat_list, X_all in [(STUFF_FEATURES, X_all_stuff), (LOCATION_FEATURES, X_all_loc), (PITCHING_FEATURES, X_all_pitch)]:
    for c in feat_list:
        med = df_train[c].median()
        X_all[c] = X_all[c].fillna(med)

pred_all_stuff = stuff_model.predict(X_all_stuff)
pred_all_loc = loc_model.predict(X_all_loc)
pred_all_pitch = pitching_model.predict(X_all_pitch)

df["Stuff+"] = compute_plus_scale(pred_all_stuff, stuff_pred_train).astype(np.float32)
df["Location+"] = compute_plus_scale(pred_all_loc, loc_pred_train).astype(np.float32)
df["Pitching+"] = compute_plus_scale(pred_all_pitch, pitch_pred_train).astype(np.float32)

# Limitar scores extremos al rango operativo esperado.
for col in ["Stuff+", "Location+", "Pitching+"]:
    df[col] = df[col].clip(60, 140)
    if not np.isfinite(df[col].to_numpy()).all():
        raise ValueError(f"{col} contiene scores no finitos despues de escalar")

print(f"\n  Scores finales en dataset completo:")
for col in ["Stuff+", "Location+", "Pitching+"]:
    print(f"    {col}: media={df[col].mean():.1f}, std={df[col].std():.1f}, rango=[{df[col].min():.1f}, {df[col].max():.1f}]")

df.to_parquet(BASE_DIR / "dataset_scores_final_v2.parquet", index=False)

# ══════════════════════════════════════════════════════════════════
# VALIDACION SEASON HOLDOUT
# ══════════════════════════════════════════════════════════════════
print(f"\n{'─' * 50}")
print("Validacion Season Holdout (train 2024-2025, test 2026)...")

df_s_train = df[df["year"].isin([2024, 2025])].copy()
df_s_test = df[df["year"] == 2026].copy()

if len(df_s_train) > 0 and len(df_s_test) > 0:
    for model_name, features, params in [
        ("Stuff+", STUFF_FEATURES, stuff_params),
        ("Location+", LOCATION_FEATURES, loc_params),
        ("Pitching+", PITCHING_FEATURES, pitch_params),
    ]:
        season_model = MODEL_CLASS(**{
            key: value for key, value in params.items()
            if key != "early_stopping_rounds"
        })
        X_train_season = df_s_train[features].copy()
        X_st = df_s_test[features].copy()
        for c in features:
            median = X_train_season[c].median()
            X_train_season[c] = X_train_season[c].fillna(median)
            X_st[c] = X_st[c].fillna(median)
        season_model.fit(X_train_season, df_s_train[TARGET])
        y_st = df_s_test[TARGET]
        pred_st = season_model.predict(X_st)
        rmse_st = np.sqrt(mean_squared_error(y_st, pred_st))
        mae_st = mean_absolute_error(y_st, pred_st)
        r2_st = r2_score(y_st, pred_st)
        validation_metrics["season_holdout"][model_name] = {
            "n": int(len(y_st)), "rmse": float(rmse_st),
            "mae": float(mae_st), "r2": float(r2_st),
        }
        print(f"  {model_name} en 2026: n={len(y_st)}, RMSE={rmse_st:.6f}, MAE={mae_st:.6f}, R2={r2_st:.4f}")

# ══════════════════════════════════════════════════════════════════
# VALIDACION POR ALTITUD
# ══════════════════════════════════════════════════════════════════
print(f"\n{'─' * 50}")
print("Validacion por Altitud...")

for cat in ["No Altitude", "Medium Altitude", "Extreme Altitude"]:
    subset = df_test[df_test["altitude_category"] == cat]
    if len(subset) > 0:
        validation_metrics["altitude_holdout"][cat] = {}
        for model_name, features, model_obj in [
            ("Stuff+", STUFF_FEATURES, stuff_model),
            ("Location+", LOCATION_FEATURES, loc_model),
            ("Pitching+", PITCHING_FEATURES, pitching_model),
        ]:
            X_sub = subset[features].copy()
            for c in features:
                X_sub[c] = X_sub[c].fillna(df_train[c].median())
            y_sub = subset[TARGET]
            pred_sub = model_obj.predict(X_sub)
            validation_metrics["altitude_holdout"][cat][model_name] = {
                "n": int(len(y_sub)),
                "rmse": float(np.sqrt(mean_squared_error(y_sub, pred_sub))),
                "mae": float(mean_absolute_error(y_sub, pred_sub)),
                "r2": float(r2_score(y_sub, pred_sub)),
            }
            print(f"  {cat} / {model_name}: n={len(y_sub)}, RMSE={validation_metrics['altitude_holdout'][cat][model_name]['rmse']:.6f}")

with open(REGISTRY_DIR / "validation_metrics.json", "w") as f:
    json.dump(validation_metrics, f, indent=2)

print(f"\n{'=' * 60}")
print("FASE 2 COMPLETADA")
print(f"{'=' * 60}")
