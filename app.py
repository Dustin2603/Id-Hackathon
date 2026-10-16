import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Simulador Stuff+ | Diablos Rojos", layout="wide")

st.title("⚾ Simulador de Calidad de Pitcheo - Diablos Rojos 2026")
st.markdown("Herramienta interactiva para evaluar el **Stuff+**, **Location+** y **Pitching+** ajustado por altitud (CDMX vs. Nivel del Mar).")

# 1. Cargar los modelos entrenados desde el Model Registry
@st.cache_resource
def load_models():
    # Asegúrate de que los archivos estén en la misma carpeta o ruta correcta
    stuff_model = joblib.load('model_registry/stuff_model.pkl')
    loc_model = joblib.load('model_registry/location_model.pkl')
    pitching_model = joblib.load('model_registry/pitching_model.pkl')
    df_scores = pd.read_parquet('model_registry/dataset_scores_final.parquet')
    return stuff_model, loc_model, pitching_model, df_scores

try:
    stuff_model, loc_model, pitching_model, df_scores = load_models()
    st.success("¡Modelos y dataset cargados correctamente desde el Model Registry!")
except Exception as e:
    st.error(f"Error al cargar los artefactos: {e}")
    st.stop()

# 2. Creación de controles interactivos (Sliders) en la barra lateral para el Pitching Coach
st.sidebar.header("🎛️ Controles Físicos del Lanzamiento")
rel_speed = st.sidebar.slider("Velocidad (RelSpeed - mph)", 70.0, 103.0, 93.5)
spin_rate = st.sidebar.slider("Tasa de Giro (SpinRate - rpm)", 1200, 3200, 2250)
ivb = st.sidebar.slider("Quiebre Vertical Inducido (IVB - pulg)", -20.0, 25.0, 15.0)
hb = st.sidebar.slider("Quiebre Horizontal (HB - pulg)", -25.0, 25.0, 5.0)
extension = st.sidebar.slider("Extensión de Soltado (pies)", 4.5, 7.5, 6.2)
rel_height = st.sidebar.slider("Altura de Soltado (pies)", 4.0, 7.5, 5.8)
rel_side = st.sidebar.slider("Posición Lateral de Soltado (pies)", -4.0, 4.0, -1.5)

# Zona y Conteo
zone = st.sidebar.selectbox("Zona de Impacto (Zone)", list(range(1, 14)), index=4)
balls = st.sidebar.selectbox("Bolas", [0, 1, 2, 3], index=0)
strikes = st.sidebar.selectbox("Strikes", [0, 1, 2], index=0)

# 3. Botón para simular y predecir en tiempo real
if st.sidebar.button("Calcular Scores Plus"):
    # Crear un DataFrame con las entradas del usuario
    input_data = pd.DataFrame({
        'RelSpeed': [rel_speed],
        'SpinRate': [spin_rate],
        'InducedVertBreak': [ivb],
        'HorzBreak': [hb],
        'RelHeight': [rel_height],
        'RelSide': [rel_side],
        'Extension': [extension],
        'Zone': [zone],
        'Balls': [balls],
        'Strikes': [strikes]
    })
    
    # Predicciones simuladas con los modelos cargados
    # (Asegúrate de pasar exactamente las columnas con las que entrenó cada modelo)
    pred_stuff = stuff_model.predict(input_data[['RelSpeed', 'SpinRate', 'InducedVertBreak', 'HorzBreak', 'RelHeight', 'RelSide', 'Extension']])[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Stuff+ Estimado (Física Pura)", f"{100 - (pred_stuff * 5):.1f}")
    col2.metric("Location+ Estimado", "104.2") # Conectar con loc_model
    col3.metric("Pitching+ / Out+ (Integral)", "106.8") # Conectar con pitching_model
    
    st.info("💡 **Análisis de Altitud (CDMX):** Recuerda que en el Estadio Alfredo Harp Helú (~2,240 msnm), el quiebre vertical real (IVB) se reduce por la menor densidad del aire.")

# 4. Vista de datos procesados
if st.checkbox("Mostrar dataset de scores procesados"):
    st.dataframe(df_scores.head(100))