import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(page_title="Telemetría LSI", layout="centered", page_icon=":D")

st.title(" Sistema Predictivo de Incrustaciones (LSI)")
st.write("Interfaz de diagnóstico utilizando un Ensamble de Inteligencia Artificial y Validación por Consenso.")

# 1. Cargar las IAs (Usamos cache para que la web no se vuelva lenta)
@st.cache_resource
def cargar_modelos():
    reg = joblib.load('ia_regresora.pkl')
    clas = joblib.load('ia_clasificadora.pkl')
    enc = joblib.load('traductor_etiquetas.pkl')
    return reg, clas, enc

try:
    ia_regresora, ia_clasificadora, traductor = cargar_modelos()
except Exception as e:
    st.error("Error: No se encontraron los modelos .pkl. Asegúrate de subirlos al repositorio.")
    st.stop()

# 2. Panel lateral para el ingreso de datos simulados
st.sidebar.header("Lecturas del Sensor")
ph = st.sidebar.number_input("pH del Agua", min_value=0.0, max_value=14.0, value=7.2, step=0.1)
cond = st.sidebar.number_input("Conductividad (µS/cm)", min_value=0.0, max_value=5000.0, value=800.0, step=10.0)
temp = st.sidebar.number_input("Temperatura (°C)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)

# 3. Procesamiento y Diagnóstico
if st.sidebar.button("Realizar Diagnóstico"):
    
    # Recrear la Ingeniería de Características matemática
    tds = cond * 0.5
    log_cond = np.log10(cond + 1)
    log_temp_k = np.log10(temp + 273.15)
    ph_x_logcond = ph * log_cond
    ph_cuadrado = ph ** 2
    cond_cuadrado = cond ** 2
    rel_termo = ph / (temp + 273.15)

    # Crear la tabla temporal para la IA
    datos_entrada = pd.DataFrame([[
        ph, cond, temp, tds, log_cond, log_temp_k, ph_x_logcond, ph_cuadrado, cond_cuadrado, rel_termo
    ]], columns=[
        'pH', 'Conductividad', 'Temperatura', 'TDS_Estimado', 'Log_Cond', 
        'Log_Temp_Kelvin', 'pH_x_LogCond', 'pH_Cuadrado', 'Cond_Cuadrado', 'Relacion_Termodinamica'
    ])

    # Lanzar predicciones
    lsi_num = ia_regresora.predict(datos_entrada)[0]
    clase_num = ia_clasificadora.predict(datos_entrada)
    clase_texto = traductor.inverse_transform(clase_num)[0]

    # Lógica de Consenso
    if lsi_num < -0.5:
        clase_reg = 'Corrosiva'
    elif lsi_num > 0.5:
        clase_reg = 'Incrustante'
    else:
        clase_reg = 'Equilibrada'

    consenso = ":D Aprobado" if clase_texto == clase_reg else ":o Revisión Manual Requerida"

    # 4. Mostrar Resultados en la Web
    st.subheader("Resultados de la Telemetría")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("LSI Matemático (IA 1)", f"{lsi_num:.2f}")
    col2.metric("Estado (IA 2)", clase_texto)
    col3.metric("Consenso de Seguridad", consenso)

    st.divider()
    
    if consenso == ":D Aprobado":
        st.success(f"**Veredicto Seguro:** El sistema de redundancia ha validado que el agua tiene propiedades de tendencia **{clase_texto}**.")
    else:
        st.warning(f"**Veredicto Dudoso:** Discrepancia detectada en los algoritmos. La IA Numérica estima un comportamiento '{clase_reg}' mientras que la IA Clasificadora lo etiqueta como '{clase_texto}'. Se recomienda un muestreo manual.")