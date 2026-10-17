"""
Fase 1 — Corrección de Datos y Features
========================================
Recalcula features relacionales del arsenal, agrega columnas faltantes,
limpia el dataset y produce `dataset_enriched.parquet`.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "Dataset Hackathon" / "stuff_model_df.parquet"
OUTPUT_PATH = BASE_DIR / "dataset_enriched.parquet"

print("=" * 60)
print("FASE 1: Corrección de Datos y Features")
print("=" * 60)

# ── 1. Cargar dataset crudo ──────────────────────────────────────
print("\n[1/6] Cargando dataset crudo...")
df = pd.read_parquet(RAW_PATH)
print(f"  Shape original: {df.shape}")

# ── 2. Limpieza básica ──────────────────────────────────────────
print("\n[2/6] Limpieza básica...")

# Eliminar filas sin pitcher_anon_id (no se pueden agrupar)
before = len(df)
df = df.dropna(subset=["pitcher_anon_id"])
print(f"  Eliminadas {before - len(df)} filas sin pitcher_anon_id")

# Eliminar filas con InducedVertBreak/HorzBreak/SpinRate nulos (702 filas)
before = len(df)
df = df.dropna(subset=["InducedVertBreak", "HorzBreak", "SpinRate"])
print(f"  Eliminadas {before - len(df)} filas sin IVB/HB/SpinRate")

# Limpiar PitcherThrows y BatterSide indefinidos
df["PitcherThrows"] = df["PitcherThrows"].fillna("Right")
df["BatterSide"] = df["BatterSide"].fillna("Right")
df.loc[df["PitcherThrows"] == "Undefined", "PitcherThrows"] = "Right"
df.loc[df["BatterSide"].isin(["Undefined", "Switch"]), "BatterSide"] = "Right"

# Limpiar altitude_category nulos
df["altitude_category"] = df["altitude_category"].fillna("No Altitude")

# Convertir EffectiveVelo a numérico (estaba como object)
df["EffectiveVelo"] = pd.to_numeric(df["EffectiveVelo"], errors="coerce")
df["EffectiveVelo"] = df["EffectiveVelo"].fillna(df["RelSpeed"])

print(f"  Shape tras limpieza: {df.shape}")

# ── 3. Features relacionales del arsenal ─────────────────────────
print("\n[3/6] Calculando features relacionales del arsenal...")

# Identificar la recta primaria de cada pitcher por temporada
fb_types = ["Four-Seam", "Sinker"]
fb_mask = df["AutoPitchType"].isin(fb_types)

# Calcular promedios de recta primaria por pitcher/año
fb_stats = (
    df[fb_mask]
    .groupby(["pitcher_anon_id", "year"])
    .agg(
        fb_avg_speed=("RelSpeed", "mean"),
        fb_avg_ivb=("InducedVertBreak", "mean"),
        fb_avg_hb=("HorzBreak", "mean"),
        fb_count=("RelSpeed", "count"),
    )
    .reset_index()
)
fb_year_stats = df[fb_mask].groupby("year")[
    ["RelSpeed", "InducedVertBreak", "HorzBreak"]
].mean().rename(columns={
    "RelSpeed": "fb_avg_speed",
    "InducedVertBreak": "fb_avg_ivb",
    "HorzBreak": "fb_avg_hb",
})
if fb_stats.empty:
    raise ValueError("No se encontraron Four-Seam o Sinker para calcular el baseline del arsenal")
fb_global_stats = df.loc[fb_mask, ["RelSpeed", "InducedVertBreak", "HorzBreak"]].mean()

print(f"  Pitchers con recta por temporada: {len(fb_stats)}")

# Merge con dataset principal
df = df.merge(fb_stats, on=["pitcher_anon_id", "year"], how="left")

# Pitchers sin recta → usar el promedio de rectas de su temporada,
# con promedio global de rectas como último fallback.
n_missing = df["fb_avg_speed"].isnull().sum()
if n_missing > 0:
    fallback_columns = {
        "fb_avg_speed": "RelSpeed",
        "fb_avg_ivb": "InducedVertBreak",
        "fb_avg_hb": "HorzBreak",
    }
    seasonal_fallbacks = {
        output: df["year"].map(fb_year_stats[input_column])
        for output, input_column in fallback_columns.items()
    }
    for output, input_column in fallback_columns.items():
        df[output] = df[output].fillna(seasonal_fallbacks[output]).fillna(
            fb_global_stats[input_column]
        )
    df["fb_count"] = df["fb_count"].fillna(0)
    print(f"  Pitchers sin recta (fallback de liga aplicado): {n_missing} lanzamientos")

# Calcular diferenciales
df["velo_diff_vs_fb"] = df["RelSpeed"] - df["fb_avg_speed"]
df["ivb_diff_vs_fb"] = df["InducedVertBreak"] - df["fb_avg_ivb"]
df["hb_diff_vs_fb"] = df["HorzBreak"] - df["fb_avg_hb"]

print(f"  velo_diff_vs_fb — media: {df['velo_diff_vs_fb'].mean():.3f}, std: {df['velo_diff_vs_fb'].std():.3f}")
print(f"  ivb_diff_vs_fb  — media: {df['ivb_diff_vs_fb'].mean():.3f}, std: {df['ivb_diff_vs_fb'].std():.3f}")
print(f"  hb_diff_vs_fb   — media: {df['hb_diff_vs_fb'].mean():.3f}, std: {df['hb_diff_vs_fb'].std():.3f}")
for col in ["velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb"]:
    if not np.isfinite(df[col].to_numpy()).all() or df[col].nunique(dropna=True) < 2:
        raise ValueError(f"La feature relacional {col} no es finita o no tiene variacion")

# ── 4. Features derivadas adicionales ───────────────────────────
print("\n[4/6] Calculando features derivadas adicionales...")

# Distancia efectiva (release al plato)
df["effective_distance"] = 60.5 - df["Extension"]

# Factor de velocidad percibida
df["perceived_velo_factor"] = (df["RelSpeed"] * 60.5) / df["effective_distance"]

# Release slot (categorización del ángulo de release)
def classify_release_slot(row):
    rel_h = row["RelHeight"]
    rel_s = abs(row["RelSide"])
    if rel_h > 6.0 and rel_s < 1.5:
        return "Over_Top"
    elif rel_h < 5.0 or rel_s > 2.5:
        return "Sidearm_Sub"
    elif rel_h >= 5.0 and rel_h <= 6.0 and rel_s >= 1.5 and rel_s <= 2.5:
        return "Three_Quarters"
    else:
        return "Standard"

df["release_slot"] = df.apply(classify_release_slot, axis=1)

# Flag CDMX
df["es_cdmx"] = df["altitude_category"] == "Extreme Altitude"

# Speed drop
df["velo_drop"] = df["RelSpeed"] - df["ZoneSpeed"]

# Encodings numéricos
df["PitcherThrows_Num"] = (df["PitcherThrows"] == "Right").astype(float)
df["BatterSide_Num"] = (df["BatterSide"] == "Right").astype(float)

print(f"  release_slot distribución: {df['release_slot'].value_counts().to_dict()}")
print(f"  es_cdmx distribución: {df['es_cdmx'].value_counts().to_dict()}")
print(f"  velo_drop media: {df['velo_drop'].mean():.2f}")

# ── 5. Construcción del target xRV ──────────────────────────────
print("\n[5/6] Construyendo target xRV...")

# Asignar run values lineales por evento (visión pitcher: menor = mejor)
# Basado en valores estándar de run expectancy
rv_map = {
    "Strikeout": -0.27,
    "StrikeSwinging": -0.065,  # no terminal, contribuye a K
    "StrikeCalled": -0.065,
    "BallCalled": 0.065,
    "BallinDirt": 0.065,
    "FoulBall": -0.038,
    "FoulBallFieldable": -0.038,
    "NeutralPlay": 0.0,
    "Out": -0.25,
    "Sacrifice": -0.15,
    "FieldersChoice": -0.15,
    "Single": 0.47,
    "Double": 0.77,
    "Triple": 1.05,
    "HomeRun": 1.40,
    "Walk": 0.33,
    "HitByPitch": 0.35,
    "Error": 0.50,
    "BallIntentional": 0.18,
}

df["xRV"] = df["play_result"].map(rv_map).fillna(0.0)

# Verificación
print(f"  xRV media: {df['xRV'].mean():.4f}")
print(f"  xRV std: {df['xRV'].std():.4f}")
print(f"  xRV rango: [{df['xRV'].min():.3f}, {df['xRV'].max():.3f}]")
print(f"  Valores sin mapear: {df['play_result'].map(rv_map).isnull().sum()}")
unmapped = df[df['play_result'].map(rv_map).isnull()]['play_result'].unique()
if len(unmapped) > 0:
    print(f"  Categorías sin mapear: {unmapped}")

# ── 6. Seleccionar columnas finales y guardar ───────────────────
print("\n[6/6] Guardando dataset enriquecido...")

# Columnas finales organizadas por categoría
cols_ids = ["PitchUID", "pitcher_anon_id", "batter_anon_id", "game_anon_id", "year"]

cols_stuff = [
    "RelSpeed", "SpinRate", "SpinAxis", "InducedVertBreak", "HorzBreak",
    "VertBreak", "EffectiveVelo", "ZoneSpeed", "SpeedDrop",
    "RelHeight", "RelSide", "Extension",
    "VertRelAngle", "HorzRelAngle", "VertApprAngle", "HorzApprAngle",
]

cols_location = ["PlateLocHeight", "PlateLocSide", "in_strike_zone", "outside_strike_zone"]

cols_context = [
    "Balls", "Strikes", "count", "Outs", "Inning",
    "PitcherThrows", "BatterSide", "AutoPitchType",
    "altitude_category",
]

cols_arsenal = [
    "fb_avg_speed", "fb_avg_ivb", "fb_avg_hb", "fb_count",
    "velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb",
]

cols_derived = [
    "effective_distance", "perceived_velo_factor", "release_slot",
    "es_cdmx", "velo_drop", "PitcherThrows_Num", "BatterSide_Num",
]

cols_outcomes = [
    "PitchCall", "KorBB", "play_result",
    "is_swing", "is_whiff", "is_contact", "is_called_strike",
    "is_swinging_strike", "is_ball_in_play",
    "ExitSpeed", "Angle", "RunsScored", "OutsOnPlay",
]

cols_target = ["xRV"]

all_cols = cols_ids + cols_stuff + cols_location + cols_context + cols_arsenal + cols_derived + cols_outcomes + cols_target

# Verificar que todas existen
missing = [c for c in all_cols if c not in df.columns]
if missing:
    print(f"  ⚠️ Columnas faltantes: {missing}")
    all_cols = [c for c in all_cols if c in df.columns]

df_final = df[all_cols].copy()
df_final.to_parquet(OUTPUT_PATH, index=False)

print(f"\n  ✅ Dataset guardado: {OUTPUT_PATH}")
print(f"  Shape final: {df_final.shape}")
print(f"  Columnas: {len(df_final.columns)}")

# ── Resumen de verificación ──────────────────────────────────────
print("\n" + "=" * 60)
print("VERIFICACIÓN FASE 1")
print("=" * 60)
print(f"\n  Features relacionales (deben ser != 0):")
print(f"    velo_diff_vs_fb std = {df_final['velo_diff_vs_fb'].std():.3f} {'✅' if df_final['velo_diff_vs_fb'].std() > 1 else '❌'}")
print(f"    ivb_diff_vs_fb  std = {df_final['ivb_diff_vs_fb'].std():.3f} {'✅' if df_final['ivb_diff_vs_fb'].std() > 1 else '❌'}")
print(f"    hb_diff_vs_fb   std = {df_final['hb_diff_vs_fb'].std():.3f} {'✅' if df_final['hb_diff_vs_fb'].std() > 1 else '❌'}")

print(f"\n  Columnas de ubicación (deben existir):")
print(f"    PlateLocHeight  ✅ media={df_final['PlateLocHeight'].mean():.2f}")
print(f"    PlateLocSide    ✅ media={df_final['PlateLocSide'].mean():.2f}")

print(f"\n  Target xRV:")
print(f"    media={df_final['xRV'].mean():.4f}, std={df_final['xRV'].std():.4f}")

print(f"\n  Nulos por columna:")
nulls = df_final.isnull().sum()
nulls_nonzero = nulls[nulls > 0]
if len(nulls_nonzero) > 0:
    for c, v in nulls_nonzero.items():
        print(f"    {c}: {v}")
else:
    print(f"    ✅ Sin nulos en ninguna columna")

print(f"\n{'=' * 60}")
print("FASE 1 COMPLETADA")
print(f"{'=' * 60}")
