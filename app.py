"""
Dashboard Stuff+ | Diablos Rojos del Mexico 2026
=================================================
Simulador interactivo de calidad de pitcheo con modelos reales,
analisis SHAP, comparacion por altitud y visualizaciones avanzadas.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import gc

# ── Configuracion de pagina ──────────────────────────────────────
st.set_page_config(
    page_title="Stuff+ Simulator | Diablos Rojos",
    page_icon="baseball",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
REGISTRY_DIR = BASE_DIR / "model_registry"
PLOTS_DIR = BASE_DIR / "plots"

# ── CSS personalizado ────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background: transparent;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #C41E3A;
    }
    .block-container { padding-top: 1rem; }
    h1 { color: #C41E3A; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════��[...]
# CARGA DE MODELOS Y DATOS
# ════════════════════════════════════════════════════════════════��[...]
@st.cache_resource
def load_models():
    """Load model artifacts from model_registry and cache them as a Streamlit resource.

    This function is decorated with @st.cache_resource so that models are loaded once
    per server process and reused across user sessions, avoiding repeated memory-heavy
    deserialization that can cause OOM on Streamlit Community Cloud.
    """
    try:
        stuff_model = joblib.load(REGISTRY_DIR / "stuff_model.pkl")
        loc_model = joblib.load(REGISTRY_DIR / "location_model.pkl")
        pitching_model = joblib.load(REGISTRY_DIR / "pitching_model.pkl")
        with open(REGISTRY_DIR / "scale_metadata.json") as f:
            scale_meta = json.load(f)

        # Run a GC pass after loading large artifacts to reduce peak memory usage.
        gc.collect()

        return stuff_model, loc_model, pitching_model, scale_meta

    except MemoryError as me:
        # Surface a clear, actionable error for OOM situations.
        raise MemoryError(
            "MemoryError while loading model artifacts. "
            "This often indicates the server doesn't have enough RAM to hold the models; "
            "consider using smaller models or upgrading the hosting plan. Details: "
            + str(me)
        )
    except FileNotFoundError as fnf:
        raise FileNotFoundError(f"Model artifact not found in model_registry: {fnf}")
    except Exception as e:
        # Re-raise as RuntimeError to make it explicit in the outer error handler.
        raise RuntimeError(f"Unexpected error loading models: {e}")


@st.cache_data
def load_data():
    candidates = [
        BASE_DIR / "dataset_scores_final_v2.parquet",
        BASE_DIR / "dataset_enriched.parquet",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("No se encontro un dataset parquet del proyecto")
    data = pd.read_parquet(path)
    for score_column in ["Stuff+", "Location+", "Pitching+"]:
        if score_column in data.columns:
            data[score_column] = pd.to_numeric(data[score_column], errors="coerce").clip(60, 140)
    return data


@st.cache_data
def load_validation_metrics():
    path = REGISTRY_DIR / "validation_metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_altitude_analysis():
    path = PLOTS_DIR / "h1_altitude_differences.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


try:
    stuff_model, loc_model, pitching_model, scale_meta = load_models()
    df_scores = load_data()
    h1_data = load_altitude_analysis()
    validation_data = load_validation_metrics()
except Exception as e:
    st.error(f"Error al cargar artefactos: {e}")
    st.stop()


def compute_plus(prediction, model_key):
    """Convierte prediccion cruda a escala Plus."""
    mu = scale_meta[model_key]["mu"]
    sigma = scale_meta[model_key]["sigma"]
    return float(np.clip(100 + 10 * (prediction - mu) / sigma, 60, 140))


def get_data_ranges():
    """Obtiene rangos P1-P99 para alertas de extrapolacion."""
    ranges = {}
    for col in ["RelSpeed", "SpinRate", "SpinAxis", "InducedVertBreak", "HorzBreak",
                 "RelHeight", "RelSide", "Extension", "PlateLocHeight", "PlateLocSide"]:
        if col in df_scores.columns:
            ranges[col] = (float(df_scores[col].quantile(0.01)), float(df_scores[col].quantile(0.99)))
    return ranges


DATA_RANGES = get_data_ranges()


def get_arsenal_reference():
    """Obtiene la referencia típica de recta usada para features relacionales."""
    source_columns = {
        "speed": ("fb_avg_speed", "RelSpeed"),
        "ivb": ("fb_avg_ivb", "InducedVertBreak"),
        "hb": ("fb_avg_hb", "HorzBreak"),
    }
    reference = {}
    fastballs = df_scores[
        df_scores.get("AutoPitchType", pd.Series(index=df_scores.index, dtype=str)).isin(
            ["Four-Seam", "Sinker"]
        )
    ] if "AutoPitchType" in df_scores.columns else df_scores.iloc[0:0]

    for key, (baseline_column, raw_column) in source_columns.items():
        if baseline_column in df_scores.columns:
            values = pd.to_numeric(df_scores[baseline_column], errors="coerce")
            if {"pitcher_anon_id", "year"}.issubset(df_scores.columns):
                values = values.loc[
                    ~df_scores.duplicated(["pitcher_anon_id", "year"])
                ]
        elif raw_column in fastballs.columns and len(fastballs):
            if {"pitcher_anon_id", "year"}.issubset(fastballs.columns):
                values = fastballs.groupby(["pitcher_anon_id", "year"])[raw_column].mean()
            else:
                values = fastballs[raw_column]
        else:
            raise ValueError(f"No hay datos de referencia para la feature de arsenal {key}")

        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            raise ValueError(f"La referencia de arsenal {key} no contiene valores validos")
        reference[key] = float(values.median())

    return reference


ARSENAL_REFERENCE = get_arsenal_reference()


def check_extrapolation(values_dict):
    """Verifica si los valores estan fuera del rango de entrenamiento."""
    warnings_list = []
    for col, val in values_dict.items():
        if col in DATA_RANGES:
            lo, hi = DATA_RANGES[col]
            if val < lo or val > hi:
                warnings_list.append(f"{col}: {val:.1f} fuera del rango [{lo:.1f}, {hi:.1f}]")
    return warnings_list


# ════════════════════════════════════════════════════════════════�[...]
# HEADER
# ════════════════════════════════════════════════════════════════�[...]
st.title("Simulador de Calidad de Pitcheo")
st.markdown("**Diablos Rojos del Mexico 2026** | Hackathon ISAC — Stuff+, Location+, Pitching+")

# ════════════════════════════════════════════════════════════════�[...]
# TABS
# ═══════════════════════════════════════════���════════════════════�[...]
tab_sim, tab_altitude, tab_explorer, tab_metrics = st.tabs([
    "Simulador", "Analisis de Altitud", "Explorador de Datos", "Metricas del Sistema"
])

# ════════════════════════════════════════════════════════════════�[...]
# TAB 1: SIMULADOR
# ════════════════════════════════════════════════════════════════�[...]
with tab_sim:
    # ── Sidebar: Controles ───────────────────────────────────────
    st.sidebar.header("Controles del Lanzamiento")

    st.sidebar.subheader("Tipo y Contexto")
    pitch_type = st.sidebar.selectbox(
        "Tipo de Pitch",
        ["Four-Seam", "Sinker", "Cutter", "Changeup", "Slider", "Curveball", "Splitter"],
        index=0
    )
    pitcher_throws = st.sidebar.selectbox("Mano del Pitcher", ["Right", "Left"], index=0)
    batter_side = st.sidebar.selectbox("Lado del Bateador", ["Right", "Left"], index=0)
    altitude = st.sidebar.selectbox(
        "Categoria de Altitud",
        ["Extreme Altitude (CDMX)", "Medium Altitude", "No Altitude"],
        index=0
    )

    st.sidebar.subheader("Fisica del Lanzamiento")
    rel_speed = st.sidebar.slider("Velocidad (RelSpeed - mph)", 70.0, 103.0, 93.5, 0.5)
    spin_rate = st.sidebar.slider("Tasa de Giro (SpinRate - rpm)", 1200, 3200, 2250, 25)
    spin_axis = st.sidebar.slider("Eje de Giro (SpinAxis - grados)", 0, 360, 200, 5)
    ivb = st.sidebar.slider("Quiebre Vertical Inducido (IVB - pulg)", -20.0, 25.0, 15.0, 0.5)
    hb = st.sidebar.slider("Quiebre Horizontal (HB - pulg)", -25.0, 25.0, 5.0, 0.5)
    extension = st.sidebar.slider("Extension (pies)", 4.5, 7.5, 6.2, 0.1)
    rel_height = st.sidebar.slider("Altura de Soltado (pies)", 4.0, 7.5, 5.8, 0.1)
    rel_side = st.sidebar.slider("Posicion Lateral (pies)", -4.0, 4.0, -1.5, 0.1)

    st.sidebar.subheader("Ubicacion y Conteo")
    plate_loc_height = st.sidebar.slider("Altura al Plato (pies)", 0.5, 4.5, 2.5, 0.1)
    plate_loc_side = st.sidebar.slider("Lateral al Plato (pies)", -2.0, 2.0, 0.0, 0.1)
    balls = st.sidebar.selectbox("Bolas", [0, 1, 2, 3], index=0)
    strikes = st.sidebar.selectbox("Strikes", [0, 1, 2], index=0)

    # ── Preparar datos de entrada ────────────────────────────────
    pitch_type_map = scale_meta["pitch_type_map"]
    pitcher_num = 1.0 if pitcher_throws == "Right" else 0.0
    batter_num = 1.0 if batter_side == "Right" else 0.0
    pitch_type_num = pitch_type_map.get(pitch_type, 8)

    # Calcular features derivadas
    eff_distance = 60.5 - extension
    eff_velo = rel_speed  # Aproximacion
    speed_drop = rel_speed * 0.07  # Aproximacion (~7%)
    vert_break = ivb - 5.0  # Aproximacion
    vert_rel_angle = -6.0 + (rel_height - 5.8) * 2  # Aproximacion
    horz_rel_angle = rel_side * 0.5  # Aproximacion

    # Features relacionales contra la referencia de arsenal del dataset.
    velo_diff = rel_speed - ARSENAL_REFERENCE["speed"]
    ivb_diff = ivb - ARSENAL_REFERENCE["ivb"]
    hb_diff = hb - ARSENAL_REFERENCE["hb"]

    in_sz = 1 if (1.5 <= plate_loc_height <= 3.5 and -0.83 <= plate_loc_side <= 0.83) else 0
    perceived_velo = (rel_speed * 60.5) / eff_distance

    # Construir DataFrames de input
    stuff_input = pd.DataFrame({
        "RelSpeed": [rel_speed], "SpinRate": [spin_rate], "SpinAxis": [spin_axis],
        "InducedVertBreak": [ivb], "HorzBreak": [hb], "VertBreak": [vert_break],
        "EffectiveVelo": [eff_velo], "SpeedDrop": [speed_drop],
        "RelHeight": [rel_height], "RelSide": [rel_side], "Extension": [extension],
        "VertRelAngle": [vert_rel_angle], "HorzRelAngle": [horz_rel_angle],
        "AutoPitchType_Num": [pitch_type_num],
        "PitcherThrows_Num": [pitcher_num], "BatterSide_Num": [batter_num],
        "velo_diff_vs_fb": [velo_diff], "ivb_diff_vs_fb": [ivb_diff], "hb_diff_vs_fb": [hb_diff],
    })

    loc_input = pd.DataFrame({
        "PlateLocHeight": [plate_loc_height], "PlateLocSide": [plate_loc_side],
        "in_strike_zone": [in_sz],
        "Balls": [balls], "Strikes": [strikes],
        "AutoPitchType_Num": [pitch_type_num],
        "PitcherThrows_Num": [pitcher_num], "BatterSide_Num": [batter_num],
    })

    pitching_input = pd.DataFrame({
        "RelSpeed": [rel_speed], "SpinRate": [spin_rate], "SpinAxis": [spin_axis],
        "InducedVertBreak": [ivb], "HorzBreak": [hb], "VertBreak": [vert_break],
        "EffectiveVelo": [eff_velo], "SpeedDrop": [speed_drop],
        "RelHeight": [rel_height], "RelSide": [rel_side], "Extension": [extension],
        "VertRelAngle": [vert_rel_angle], "HorzRelAngle": [horz_rel_angle],
        "PlateLocHeight": [plate_loc_height], "PlateLocSide": [plate_loc_side],
        "in_strike_zone": [in_sz],
        "Balls": [balls], "Strikes": [strikes],
        "AutoPitchType_Num": [pitch_type_num],
        "PitcherThrows_Num": [pitcher_num], "BatterSide_Num": [batter_num],
        "velo_diff_vs_fb": [velo_diff], "ivb_diff_vs_fb": [ivb_diff], "hb_diff_vs_fb": [hb_diff],
        "effective_distance": [eff_distance], "perceived_velo_factor": [perceived_velo],
    })

    # ── Predicciones ─────────────────────────────────────────────
    pred_stuff = stuff_model.predict(stuff_input[scale_meta["features"]["stuff"]])[0]
    pred_loc = loc_model.predict(loc_input[scale_meta["features"]["location"]])[0]
    pred_pitch = pitching_model.predict(pitching_input[scale_meta["features"]["pitching"]])[0]

    stuff_plus = compute_plus(pred_stuff, "stuff")
    loc_plus = compute_plus(pred_loc, "location")
    pitch_plus = compute_plus(pred_pitch, "pitching")

    # ── Alertas de extrapolacion ─────────────────────────────────
    extrap_checks = {
        "RelSpeed": rel_speed, "SpinRate": spin_rate, "InducedVertBreak": ivb,
        "HorzBreak": hb, "RelHeight": rel_height, "RelSide": rel_side,
        "Extension": extension, "PlateLocHeight": plate_loc_height,
        "PlateLocSide": plate_loc_side,
    }
    extrap_warnings = check_extrapolation(extrap_checks)

    # ── Mostrar resultados ───────────────────────────────────────
    st.subheader(f"Resultados: {pitch_type} | {pitcher_throws} | Conteo {balls}-{strikes}")

    col1, col2, col3 = st.columns(3)

    def score_delta(val):
        return f"{val - 100:+.1f}"

    col1.metric("Stuff+ (Fisica Pura)", f"{stuff_plus:.1f}", delta=score_delta(stuff_plus))
    col2.metric("Location+ (Ubicacion)", f"{loc_plus:.1f}", delta=score_delta(loc_plus))
    col3.metric("Pitching+ (Integral)", f"{pitch_plus:.1f}", delta=score_delta(pitch_plus))

    if extrap_warnings:
        st.warning("**Alerta de extrapolacion:** Los siguientes valores estan fuera del rango de entrenamiento:\n" +
                   "\n".join([f"- {w}" for w in extrap_warnings]))

    # ── Grafico de movimiento ────────────────────────────────────
    st.subheader("Mapa de Movimiento del Pitch")
    col_mov, col_zone = st.columns(2)

    with col_mov:
        # Scatter de movimiento con el pitch simulado
        fig_mov = go.Figure()

        # Datos de referencia por tipo
        main_types = ["Four-Seam", "Sinker", "Slider", "Changeup", "Curveball", "Cutter"]
        colors_map = {
            "Four-Seam": "#FF6B6B", "Sinker": "#4ECDC4", "Slider": "#FFE66D",
            "Changeup": "#95E1D3", "Curveball": "#AA96DA", "Cutter": "#F38181"
        }

        if "AutoPitchType" in df_scores.columns:
            for pt in main_types:
                sub = df_scores[df_scores["AutoPitchType"] == pt].sample(
                    min(500, len(df_scores[df_scores["AutoPitchType"] == pt])), random_state=42
                )
                fig_mov.add_trace(go.Scatter(
                    x=sub["HorzBreak"], y=sub["InducedVertBreak"],
                    mode="markers", name=pt, opacity=0.2,
                    marker=dict(size=4, color=colors_map.get(pt, "gray")),
                ))

        # Pitch simulado
        fig_mov.add_trace(go.Scatter(
            x=[hb], y=[ivb],
            mode="markers+text", name="Tu Pitch",
            marker=dict(size=18, color="#C41E3A", symbol="diamond", line=dict(width=2, color="white")),
            text=[f"{pitch_type}"], textposition="top center",
            textfont=dict(size=12, color="white"),
        ))

        fig_mov.update_layout(
            xaxis_title="Horizontal Break (pulg)",
            yaxis_title="IVB (pulg)",
            xaxis=dict(range=[-25, 25], zeroline=True),
            yaxis=dict(range=[-25, 25], zeroline=True),
            height=450,
            template="plotly_dark",
            showlegend=True,
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig_mov, use_container_width=True)

    with col_zone:
        # Strike zone con ubicacion del pitch
        fig_zone = go.Figure()

        # Strike zone box
        fig_zone.add_shape(
            type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
            line=dict(color="white", width=2),
        )
        # Home plate
        fig_zone.add_shape(
            type="path",
            path="M -0.71 0.5 L 0.71 0.5 L 0.71 0.3 L 0 0 L -0.71 0.3 Z",
            line=dict(color="white", width=1),
            fillcolor="rgba(255,255,255,0.1)",
        )

        # Pitch location
        color_zone = "#2ECC71" if in_sz else "#E74C3C"
        fig_zone.add_trace(go.Scatter(
            x=[plate_loc_side], y=[plate_loc_height],
            mode="markers",
            marker=dict(size=25, color=color_zone, line=dict(width=2, color="white")),
            name="Ubicacion",
        ))

        fig_zone.update_layout(
            xaxis_title="Lateral (pies)",
            yaxis_title="Altura (pies)",
            xaxis=dict(range=[-2.5, 2.5], zeroline=False),
            yaxis=dict(range=[-0.5, 5.0], zeroline=False),
            height=450,
            template="plotly_dark",
            showlegend=False,
            title="Zona de Strike",
        )
        st.plotly_chart(fig_zone, use_container_width=True)

    # ── Interpretacion SHAP ──────────────────────────────────────
    st.subheader("Explicacion del Score (SHAP)")
    try:
        import shap

        shap_col1, shap_col2, shap_col3 = st.columns(3)

        with shap_col1:
            st.markdown("**Stuff+ - Factores principales**")
            explainer_s = shap.TreeExplainer(stuff_model)
            shap_vals_s = explainer_s.shap_values(stuff_input[scale_meta["features"]["stuff"]])

            shap_df = pd.DataFrame({
                "Feature": scale_meta["features"]["stuff"],
                "SHAP Value": shap_vals_s[0],
                "Valor": stuff_input[scale_meta["features"]["stuff"]].values[0],
            }).sort_values("SHAP Value", key=abs, ascending=False).head(10)

            fig_shap = go.Figure()
            colors_shap = ["#2ECC71" if v > 0 else "#E74C3C" for v in shap_df["SHAP Value"]]
            fig_shap.add_trace(go.Bar(
                y=shap_df["Feature"][::-1],
                x=shap_df["SHAP Value"][::-1],
                orientation="h",
                marker_color=colors_shap[::-1],
                text=[f"{v:.4f}" for v in shap_df["SHAP Value"][::-1]],
                textposition="outside",
            ))
            fig_shap.update_layout(
                height=350, template="plotly_dark",
                xaxis_title="Contribucion SHAP",
                margin=dict(l=0, r=50, t=10, b=30),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        with shap_col2:
            st.markdown("**Pitching+ - Factores principales**")
            explainer_p = shap.TreeExplainer(pitching_model)
            shap_vals_p = explainer_p.shap_values(pitching_input[scale_meta["features"]["pitching"]])

            shap_df_p = pd.DataFrame({
                "Feature": scale_meta["features"]["pitching"],
                "SHAP Value": shap_vals_p[0],
            }).sort_values("SHAP Value", key=abs, ascending=False).head(10)

            fig_shap_p = go.Figure()
            colors_shap_p = ["#2ECC71" if v > 0 else "#E74C3C" for v in shap_df_p["SHAP Value"]]
            fig_shap_p.add_trace(go.Bar(
                y=shap_df_p["Feature"][::-1],
                x=shap_df_p["SHAP Value"][::-1],
                orientation="h",
                marker_color=colors_shap_p[::-1],
                text=[f"{v:.4f}" for v in shap_df_p["SHAP Value"][::-1]],
                textposition="outside",
            ))
            fig_shap_p.update_layout(
                height=350, template="plotly_dark",
                xaxis_title="Contribucion SHAP",
                margin=dict(l=0, r=50, t=10, b=30),
            )
            st.plotly_chart(fig_shap_p, use_container_width=True)

        with shap_col3:
            st.markdown("**Location+ - Factores principales**")
            explainer_l = shap.TreeExplainer(loc_model)
            shap_vals_l = explainer_l.shap_values(loc_input[scale_meta["features"]["location"]])
            shap_df_l = pd.DataFrame({
                "Feature": scale_meta["features"]["location"],
                "SHAP Value": shap_vals_l[0],
            }).sort_values("SHAP Value", key=abs, ascending=False).head(10)

            fig_shap_l = go.Figure()
            colors_shap_l = ["#2ECC71" if value > 0 else "#E74C3C"
                             for value in shap_df_l["SHAP Value"]]
            fig_shap_l.add_trace(go.Bar(
                y=shap_df_l["Feature"][::-1],
                x=shap_df_l["SHAP Value"][::-1],
                orientation="h",
                marker_color=colors_shap_l[::-1],
                text=[f"{value:.4f}" for value in shap_df_l["SHAP Value"][::-1]],
                textposition="outside",
            ))
            fig_shap_l.update_layout(
                height=350, template="plotly_dark",
                xaxis_title="Contribucion SHAP",
                margin=dict(l=0, r=50, t=10, b=30),
            )
            st.plotly_chart(fig_shap_l, use_container_width=True)

    except ImportError:
        st.info("Instala `shap` para ver la explicacion del score: `pip install shap`")
    except Exception as e:
        st.warning(f"No se pudo calcular SHAP: {e}")

# ════════════════════════════════════════════════════════════════�[...]
# TAB 2: ANALISIS DE ALTITUD
# ════════════════════════════════════════════════════════════════�[...]
with tab_altitude:
    st.subheader("Comparacion CDMX (2,240 msnm) vs. Nivel del Mar")
    st.markdown("""
    El Estadio Alfredo Harp Helu se ubica a ~2,240 msnm. La menor densidad del aire
    puede modificar la resistencia aerodinamica y el efecto Magnus, afectando el movimiento
    del pitch.
    """)

    if h1_data is not None:
        col_h1, col_h2 = st.columns(2)

        with col_h1:
            st.markdown("**Delta IVB (CDMX - No Altitude) por tipo**")
            h1_ivb = h1_data[h1_data["metric"] == "InducedVertBreak"]
            fig_h1 = go.Figure()
            colors_h1 = ["#C41E3A" if p < 0.05 else "#888888" for p in h1_ivb["p_value"]]
            fig_h1.add_trace(go.Bar(
                x=h1_ivb["pitch_type"], y=h1_ivb["delta"],
                marker_color=colors_h1,
                text=[f"{d:+.2f}" for d in h1_ivb["delta"]],
                textposition="outside",
            ))
            fig_h1.update_layout(
                yaxis_title="Delta IVB (pulg)",
                height=400, template="plotly_dark",
            )
            fig_h1.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            st.plotly_chart(fig_h1, use_container_width=True)

        with col_h2:
            st.markdown("**Delta HB (CDMX - No Altitude) por tipo**")
            h1_hb = h1_data[h1_data["metric"] == "HorzBreak"]
            fig_h2 = go.Figure()
            colors_h2 = ["#C41E3A" if p < 0.05 else "#888888" for p in h1_hb["p_value"]]
            fig_h2.add_trace(go.Bar(
                x=h1_hb["pitch_type"], y=h1_hb["delta"],
                marker_color=colors_h2,
                text=[f"{d:+.2f}" for d in h1_hb["delta"]],
                textposition="outside",
            ))
            fig_h2.update_layout(
                yaxis_title="Delta HB (pulg)",
                height=400, template="plotly_dark",
            )
            fig_h2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            st.plotly_chart(fig_h2, use_container_width=True)

        # Tabla completa
        st.subheader("Tabla de Diferencias Detallada")
        st.dataframe(
            h1_data.style.format({
                "cdmx_mean": "{:.2f}", "noalt_mean": "{:.2f}",
                "delta": "{:+.3f}", "p_value": "{:.4f}",
            }),
            use_container_width=True,
        )
    else:
        st.info("Ejecuta `analyze_shap.py` para generar el analisis de altitud.")

    # Imagenes SHAP
    st.subheader("Stuff+ por Altitud y Tipo de Pitch")
    img_path = PLOTS_DIR / "stuff_plus_by_altitude_type.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

    st.subheader("Mapa de Movimiento General")
    img_path2 = PLOTS_DIR / "pitch_movement_map.png"
    if img_path2.exists():
        st.image(str(img_path2), use_container_width=True)


# ════════════════════════════════════════════════════════════════�[...]
# TAB 3: EXPLORADOR DE DATOS
# ════════════════════════════════════════════════════════════════�[...]
with tab_explorer:
    st.subheader("Explorador de Datos y Perfiles")

    # Scores por pitcher
    if "pitcher_anon_id" in df_scores.columns and "Stuff+" in df_scores.columns:
        st.markdown("**Scores Promedio por Pitcher (Top 30)**")

        plus_cols = [c for c in ["Stuff+", "Location+", "Pitching+"] if c in df_scores.columns]
        pitcher_stats = (
            df_scores.groupby("pitcher_anon_id")
            .agg(
                n_pitches=("RelSpeed", "count"),
                avg_velo=("RelSpeed", "mean"),
                avg_spin=("SpinRate", "mean"),
                **{f"avg_{c}": (c, "mean") for c in plus_cols},
            )
            .reset_index()
        )
        pitcher_stats = pitcher_stats[pitcher_stats["n_pitches"] >= 100]
        pitcher_stats = pitcher_stats.sort_values(f"avg_{plus_cols[-1]}", ascending=False).head(30)

        st.dataframe(
            pitcher_stats.style.format({
                "avg_velo": "{:.1f}", "avg_spin": "{:.0f}",
                **{f"avg_{c}": "{:.1f}" for c in plus_cols},
            }),
            use_container_width=True,
        )

    # Distribucion de scores
    st.markdown("**Distribucion de Scores Plus**")
    score_col = st.selectbox("Score a visualizar", ["Stuff+", "Location+", "Pitching+"])
    if score_col in df_scores.columns:
        fig_dist = px.histogram(
            df_scores, x=score_col, nbins=80,
            color_discrete_sequence=["#C41E3A"],
            template="plotly_dark",
        )
        fig_dist.add_vline(x=100, line_dash="dash", line_color="white", annotation_text="Promedio (100)")
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Datos raw
    if st.checkbox("Mostrar datos crudos (primeras 200 filas)"):
        st.dataframe(df_scores.head(200), use_container_width=True)


# ════════════════════════════════════════════════════════════════�[...]
# TAB 4: METRICAS DEL SISTEMA
# ════════════════════════════════════════════════════════════════�[...]
with tab_metrics:
    st.subheader("Metricas de Validacion del Sistema")

    if validation_data:
        st.markdown("**Holdout por pitcher**")
        pitcher_rows = [
            {"Modelo": name, "N": result["n"], "RMSE": result["rmse"],
             "MAE": result["mae"], "R2": result["r2"]}
            for name, result in validation_data.get("pitcher_holdout", {}).items()
        ]
        if pitcher_rows:
            st.dataframe(pd.DataFrame(pitcher_rows).style.format(
                {"RMSE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:.4f}"}
            ), use_container_width=True)

        st.markdown("**Holdout estacional: entrenamiento 2024-2025, prueba 2026**")
        season_rows = [
            {"Modelo": name, "N": result["n"], "RMSE": result["rmse"],
             "MAE": result["mae"], "R2": result["r2"]}
            for name, result in validation_data.get("season_holdout", {}).items()
        ]
        if season_rows:
            st.dataframe(pd.DataFrame(season_rows).style.format(
                {"RMSE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:.4f}"}
            ), use_container_width=True)

        st.markdown("**Holdout por altitud (pitcher)**")
        altitude_rows = [
            {"Altitud": category, "Modelo": name, "N": result["n"],
             "RMSE": result["rmse"], "MAE": result["mae"], "R2": result["r2"]}
            for category, models in validation_data.get("altitude_holdout", {}).items()
            for name, result in models.items()
        ]
        if altitude_rows:
            st.dataframe(pd.DataFrame(altitude_rows).style.format(
                {"RMSE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:.4f}"}
            ), use_container_width=True)
    else:
        st.warning("No hay metricas de validacion generadas. Ejecuta train_models.py.")

    # SHAP plots
    st.subheader("Feature Importance (SHAP)")
    shap_tabs = st.tabs(["Stuff+", "Location+", "Pitching+"])
    for i, name in enumerate(["Stuffplus", "Locationplus", "Pitchingplus"]):
        with shap_tabs[i]:
            for plot_type in ["shap_global", "shap_beeswarm"]:
                img = PLOTS_DIR / f"{plot_type}_{name}.png"
                if img.exists():
                    st.image(str(img), use_container_width=True)

    st.subheader("Informacion del Dataset")
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Total Lanzamientos", f"{len(df_scores):,}")
    if "pitcher_anon_id" in df_scores.columns:
        col_info2.metric("Pitchers Unicos", f"{df_scores['pitcher_anon_id'].nunique():,}")
    if "year" in df_scores.columns:
        col_info3.metric("Temporadas", f"{df_scores['year'].nunique()}")

    if "altitude_category" in df_scores.columns:
        st.markdown("**Distribucion por Altitud**")
        alt_counts = df_scores["altitude_category"].value_counts()
        fig_alt = px.pie(
            values=alt_counts.values, names=alt_counts.index,
            color_discrete_sequence=["#2196F3", "#FF9800", "#C41E3A"],
            template="plotly_dark",
        )
        fig_alt.update_layout(height=350)
        st.plotly_chart(fig_alt, use_container_width=True)
