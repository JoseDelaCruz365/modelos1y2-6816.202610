import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelos/modelo_random_forest.pkl")

modelo = cargar_modelo()

# =========================
# Interfaz de la aplicación
# =========================
st.set_page_config(
    page_title="Predicción de Diabetes",
    layout="centered"
)

st.title("🩺 Predicción de Diabetes")
st.write("El modelo espera:", modelo.n_features_in_, "variables")
st.write(
    "Ingrese los datos clínicos del paciente para predecir "
    "la presencia o ausencia de diabetes."
)

# =========================
# Entradas del usuario
# =========================
st.subheader("📋 Variables clínicas")

embarazos = st.number_input("Número de embarazos", 0, 20, 1)
glucosa = st.number_input("Nivel de glucosa", 0.0, 300.0, 120.0)
presion = st.number_input("Presión arterial", 0.0, 200.0, 70.0)
pliegue = st.number_input("Pliegue cutáneo", 0.0, 100.0, 20.0)
insulina = st.number_input("Insulina", 0.0, 900.0, 80.0)
imc = st.number_input("Índice de masa corporal (IMC)", 0.0, 70.0, 25.0)
pedigri = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
edad = st.number_input("Edad", 0, 120, 30)


# =========================
# Predicción
# =========================
if st.button("🔍 Realizar predicción"):
    datos = np.array([[embarazos, glucosa, presion, pliegue,
                       insulina, imc, pedigri, edad]])

    prediccion = modelo.predict(datos)[0]

    st.subheader("📊 Resultado")
    if prediccion == 1:
        st.error("⚠️ El modelo predice PRESENCIA de diabetes")
    else:
        st.success("✅ El modelo predice AUSENCIA de diabetes")

# =========================
# Información del estudiante
# =========================
st.markdown("---")
st.markdown("### 👨‍🎓 Información del estudiante")
st.write("**Nombre:** JOSE MIGUEL DE LA CRUZ GAMARRA")
st.write("**Código ISIL:** 73083425")
st.markdown(
    "**Cuaderno en Google Colab:** "
    "[Abrir notebook](https://colab.research.google.com/drive/1SSDKd3iUkqNtUzMGDZ9i3HFMvTq5_xkl?usp=sharing)"
)
