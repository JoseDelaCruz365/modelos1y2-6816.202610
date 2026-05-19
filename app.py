import streamlit as st
import joblib
import numpy as np

# =========================
# Cargar el modelo
# =========================
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_diabetes.pkl")

modelo = cargar_modelo()

# =========================
# Interfaz de la aplicación
# =========================
st.set_page_config(
    page_title="Predicción de Diabetes",
    layout="centered"
)

st.title("🩺 Predicción de Diabetes")
st.write(
    "Ingrese los datos clínicos del paciente para predecir "
    "la presencia o ausencia de diabetes."
)

# =========================
# Entradas del usuario
# =========================
st.subheader("📋 Variables clínicas")

glucosa = st.number_input(
    "Nivel de glucosa (mg/dL)",
    min_value=0.0,
    max_value=300.0,
    value=120.0
)

presion = st.number_input(
    "Presión arterial (mm Hg)",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

imc = st.number_input(
    "Índice de Masa Corporal (IMC)",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

edad = st.number_input(
    "Edad (años)",
    min_value=0,
    max_value=120,
    value=30
)

# =========================
# Predicción
# =========================
if st.button("🔍 Realizar predicción"):
    datos = np.array([[glucosa, presion, imc, edad]])
    prediccion = modelo.predict(datos)[0]

    st.subheader("📊 Resultado")

    if prediccion == 1:
        st.error("⚠️ El modelo predice **PRESENCIA de diabetes**")
    else:
        st.success("✅ El modelo predice **AUSENCIA de diabetes**")

# =========================
# Información del estudiante
# =========================
st.markdown("---")
st.markdown("### 👨‍🎓 Información del estudiante")
st.write("**Nombre:** Juan Pérez")
st.write("**Código ISIL:** ISIL20240001")
st.markdown(
    "**Cuaderno en Google Colab:** "
    "[Abrir notebook](https://colab.research.google.com/)"
)
