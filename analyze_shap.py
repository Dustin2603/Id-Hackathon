"""
Fase 3 — Analisis SHAP y Altitud
==================================
Genera graficos SHAP, analisis de altitud y exporta artefactos para el dashboard.
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
REGISTRY_DIR = BASE_DIR / "model_registry"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("FASE 3: Analisis SHAP y Altitud")
print("=" * 60)

# ── Cargar modelos y datos ──────────────────────────────────────
print("\nCargando modelos y datos...")
stuff_model = joblib.load(REGISTRY_DIR / "stuff_model.pkl")
loc_model = joblib.load(REGISTRY_DIR / "location_model.pkl")
pitching_model = joblib.load(REGISTRY_DIR / "pitching_model.pkl")

with open(REGISTRY_DIR / "scale_metadata.json") as f:
    meta = json.load(f)

df = pd.read_parquet(BASE_DIR / "dataset_enriched.parquet")

# Preparar datos
pitch_type_map = meta["pitch_type_map"]
df["AutoPitchType_Num"] = df["AutoPitchType"].map(pitch_type_map).fillna(8).astype(int)
df["Extension"] = df["Extension"].fillna(df["Extension"].median())
df["effective_distance"] = 60.5 - df["Extension"]
df["perceived_velo_factor"] = (df["RelSpeed"] * 60.5) / df["effective_distance"]
df["fb_count"] = df["fb_count"].fillna(0)
df["SpinAxis"] = df["SpinAxis"].fillna(df["SpinAxis"].median())
df["SpeedDrop"] = df["SpeedDrop"].fillna(df["RelSpeed"] - df["ZoneSpeed"])

STUFF_FEATURES = meta["features"]["stuff"]
LOCATION_FEATURES = meta["features"]["location"]
PITCHING_FEATURES = meta["features"]["pitching"]

# Muestra para SHAP (usar 5000 para velocidad)
np.random.seed(42)
sample_idx = np.random.choice(len(df), min(5000, len(df)), replace=False)
df_sample = df.iloc[sample_idx].copy()

# ══════════════════════════════════════════════════════════════════
# SHAP GLOBAL PARA CADA MODELO
# ══════════════════════════════════════════════════════════════════
print("\n[1/4] Calculando SHAP values...")

shap_results = {}
for model_name, features, model in [
    ("Stuff+", STUFF_FEATURES, stuff_model),
    ("Location+", LOCATION_FEATURES, loc_model),
    ("Pitching+", PITCHING_FEATURES, pitching_model),
]:
    print(f"  Procesando {model_name}...")
    X_sample = df_sample[features].copy()
    for c in features:
        X_sample[c] = X_sample[c].fillna(df[c].median())
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap_results[model_name] = (X_sample, shap_values, features)
    
    # Bar plot (importancia global)
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    top_n = min(15, len(features))
    ax.barh(
        range(top_n),
        mean_abs_shap[sorted_idx[:top_n]][::-1],
        color="#C41E3A"
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([features[i] for i in sorted_idx[:top_n]][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"{model_name} - Feature Importance (SHAP)")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"shap_global_{model_name.replace('+','plus')}.png", dpi=150)
    plt.close(fig)
    print(f"    Guardado: shap_global_{model_name.replace('+','plus')}.png")
    
    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=features, show=False, max_display=15)
    plt.title(f"{model_name} - SHAP Beeswarm")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"shap_beeswarm_{model_name.replace('+','plus')}.png", dpi=150)
    plt.close('all')
    print(f"    Guardado: shap_beeswarm_{model_name.replace('+','plus')}.png")

# ══════════════════════════════════════════════════════════════════
# ANALISIS DE ALTITUD (Hipotesis H1-H4)
# ══════════════════════════════════════════════════════════════════
print("\n[2/4] Analisis de altitud (H1-H4)...")

cdmx = df[df["altitude_category"] == "Extreme Altitude"]
no_alt = df[df["altitude_category"] == "No Altitude"]
med_alt = df[df["altitude_category"] == "Medium Altitude"]

print(f"  Extreme Altitude (CDMX): {len(cdmx)}")
print(f"  No Altitude: {len(no_alt)}")
print(f"  Medium Altitude: {len(med_alt)}")

# H1: Diferencias de IVB/HB por tipo de pitch entre categorias
print("\n  --- H1: Diferencias fisicas por tipo y altitud ---")
main_types = ["Four-Seam", "Sinker", "Slider", "Changeup", "Curveball", "Cutter"]

h1_results = []
for pt in main_types:
    cdmx_pt = cdmx[cdmx["AutoPitchType"] == pt]
    noalt_pt = no_alt[no_alt["AutoPitchType"] == pt]
    
    if len(cdmx_pt) > 30 and len(noalt_pt) > 30:
        for metric in ["InducedVertBreak", "HorzBreak", "RelSpeed"]:
            delta = cdmx_pt[metric].mean() - noalt_pt[metric].mean()
            t_stat, p_val = stats.ttest_ind(cdmx_pt[metric].dropna(), noalt_pt[metric].dropna())
            
            h1_results.append({
                "pitch_type": pt,
                "metric": metric,
                "cdmx_mean": cdmx_pt[metric].mean(),
                "noalt_mean": noalt_pt[metric].mean(),
                "delta": delta,
                "p_value": p_val,
                "n_cdmx": len(cdmx_pt),
                "n_noalt": len(noalt_pt),
            })
            sig = "*" if p_val < 0.05 else ""
            print(f"    {pt:12s} {metric:20s}: delta={delta:+.3f} p={p_val:.4f}{sig}")

h1_df = pd.DataFrame(h1_results)
h1_df.to_csv(PLOTS_DIR / "h1_altitude_differences.csv", index=False)

# H1 Plot: Delta IVB por tipo
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, metric, title in [
    (axes[0], "InducedVertBreak", "Delta IVB (CDMX - No Altitude)"),
    (axes[1], "HorzBreak", "Delta HB (CDMX - No Altitude)"),
]:
    h1_sub = h1_df[h1_df["metric"] == metric]
    colors = ["#C41E3A" if p < 0.05 else "#888888" for p in h1_sub["p_value"]]
    ax.barh(h1_sub["pitch_type"], h1_sub["delta"], color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Delta (pulgadas)" if metric != "RelSpeed" else "Delta (mph)")
    ax.set_title(title)

plt.tight_layout()
fig.savefig(PLOTS_DIR / "h1_altitude_deltas.png", dpi=150)
plt.close(fig)
print("    Guardado: h1_altitude_deltas.png")

# ══════════════════════════════════════════════════════════════════
# SHAP POR ALTITUD
# ══════════════════════════════════════════════════════════════════
print("\n[3/4] SHAP por altitud...")

X_stuff_all = df[STUFF_FEATURES].copy()
for c in STUFF_FEATURES:
    X_stuff_all[c] = X_stuff_all[c].fillna(df[c].median())

# Calcular Stuff+ predicciones
stuff_preds = stuff_model.predict(X_stuff_all)
mu_s, sigma_s = meta["stuff"]["mu"], meta["stuff"]["sigma"]
df["Stuff+_calc"] = np.clip(100 + 10 * (stuff_preds - mu_s) / sigma_s, 60, 140)

print("\n  Stuff+ promedio por altitud y tipo:")
pivot = df.groupby(["altitude_category", "AutoPitchType"])["Stuff+_calc"].agg(["mean", "count"]).reset_index()
pivot_main = pivot[pivot["AutoPitchType"].isin(main_types)]
pivot_table = pivot_main.pivot(index="AutoPitchType", columns="altitude_category", values="mean")
print(pivot_table.round(1).to_string())

pivot_table.to_csv(PLOTS_DIR / "stuff_plus_by_altitude_type.csv")

# Plot: Stuff+ por altitud y tipo
fig, ax = plt.subplots(figsize=(12, 6))
pivot_table.plot(kind="bar", ax=ax, color=["#2196F3", "#FF9800", "#C41E3A"])
ax.set_ylabel("Stuff+ Promedio")
ax.set_title("Stuff+ por Tipo de Pitch y Categoria de Altitud")
ax.axhline(y=100, color="black", linestyle="--", alpha=0.5, label="Promedio (100)")
ax.legend(title="Altitud")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "stuff_plus_by_altitude_type.png", dpi=150)
plt.close(fig)
print("  Guardado: stuff_plus_by_altitude_type.png")

# ══════════════════════════════════════════════════════════════════
# PITCH MOVEMENT PLOT
# ══════════════════════════════════════════════════════════════════
print("\n[4/4] Generando grafico de movimiento...")

fig, ax = plt.subplots(figsize=(10, 10))
for pt in main_types:
    sub = df[df["AutoPitchType"] == pt].sample(min(2000, len(df[df["AutoPitchType"] == pt])), random_state=42)
    ax.scatter(sub["HorzBreak"], sub["InducedVertBreak"], alpha=0.3, s=8, label=pt)

ax.set_xlabel("Horizontal Break (pulg)")
ax.set_ylabel("Induced Vertical Break (pulg)")
ax.set_title("Mapa de Movimiento por Tipo de Pitch")
ax.axhline(y=0, color="gray", linewidth=0.5)
ax.axvline(x=0, color="gray", linewidth=0.5)
ax.legend(markerscale=3)
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "pitch_movement_map.png", dpi=150)
plt.close(fig)
print("  Guardado: pitch_movement_map.png")

print("\n" + "=" * 60)
print("FASE 3 COMPLETADA")
print("=" * 60)
print(f"\nArchivos generados en {PLOTS_DIR}:")
for f in sorted(PLOTS_DIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size // 1024} KB)")
