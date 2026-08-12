import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(page_title="Telemetria LSI", layout="wide", page_icon=":D")

st.title("Sistema Predictivo de Incrustaciones (LSI)")
st.write("Interfaz de diagnostico utilizando telemetria IoT y un Ensamble de Inteligencia Artificial.")

# ====================================================================
# 1. CARGA DE MODELOS
# ====================================================================
@st.cache_resource
def cargar_modelos():
    reg = joblib.load('ia_regresora.pkl')
    clas = joblib.load('ia_clasificadora.pkl')
    enc = joblib.load('traductor_etiquetas.pkl')
    return reg, clas, enc

try:
    ia_regresora, ia_clasificadora, traductor = cargar_modelos()
except Exception as e:
    st.error("Error: No se encontraron los modelos .pkl en el repositorio.")
    st.stop()

# ====================================================================
# 2. CONEXIÓN A GOOGLE SHEETS
# ====================================================================
# La URL de tu archivo en tiempo real
URL_GOOGLE_SHEETS = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQN1fUNX4RW58QQBpT7yUivdAxivFiYpPsf9m7XNLE9ubdjSwcq6wYSuB62xSg6YwLl0r8ChcSzDjo8/pub?output=csv"

# Funcion para leer los datos, se refresca cada 10 segundos
@st.cache_data(ttl=10)
def obtener_datos():
    # Leer el archivo indicando que no hay encabezados si es puro dato, 
    # o leyendo la primera fila si tiene nombres. 
    # Asumimos que la columna 0 es tiempo, 1 es Temp, 2 es pH, 3 es Cond.
    return pd.read_csv(URL_GOOGLE_SHEETS)

st.sidebar.header("Origen de Datos")
modo_conexion = st.sidebar.radio("Selecciona el modo:", ["Telemetria (Google Sheets)", "Simulacion Manual"])

# Variables globales vacias
ph, cond, temp = 0.0, 0.0, 0.0
fecha_lectura = "N/A"

if modo_conexion == "Telemetria (Google Sheets)":
    st.sidebar.success("Conectado a la base de datos de la UTEQ.")
    
    try:
        df_telemetria = obtener_datos()
        
        # Extraemos la ultima fila (la mas reciente)
        ultima_lectura = df_telemetria.iloc[-1]
        
        # Las columnas segun lo que especificaste:
        # Columna 0 (A): Tiempo
        # Columna 1 (B): Temperatura
        # Columna 2 (C): pH
        # Columna 3 (D): Conductividad
        fecha_lectura = str(ultima_lectura.iloc[0])
        temp = float(ultima_lectura.iloc[1])
        ph = float(ultima_lectura.iloc[2])
        cond = float(ultima_lectura.iloc[3])
        
        # Mostramos los datos extraidos en el lateral
        st.sidebar.markdown(f"**Ultima actualizacion:** {fecha_lectura}")
        st.sidebar.metric("pH", f"{ph}")
        st.sidebar.metric("Conductividad", f"{cond} uS/cm")
        st.sidebar.metric("Temperatura", f"{temp} C")
        
    except Exception as e:
        st.sidebar.error("Error al leer el archivo. Revisa el formato de las columnas.")
        
else:
    st.sidebar.subheader("Lecturas Manuales")
    ph = st.sidebar.number_input("pH del Agua", min_value=0.0, max_value=14.0, value=7.2, step=0.1)
    cond = st.sidebar.number_input("Conductividad (uS/cm)", min_value=0.0, max_value=5000.0, value=800.0, step=10.0)
    temp = st.sidebar.number_input("Temperatura (C)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)

# ====================================================================
# 3. PROCESAMIENTO E INTELIGENCIA ARTIFICIAL
# ====================================================================
# Ejecucion automatica si es Telemetria, o al presionar el boton si es manual.
if st.button("Realizar Diagnostico") or modo_conexion == "Telemetria (Google Sheets)":
    
    # Recrear la Ingenieria de Caracteristicas
    tds = cond * 0.5
    log_cond = np.log10(cond + 1)
    log_temp_k = np.log10(temp + 273.15)
    ph_x_logcond = ph * log_cond
    ph_cuadrado = ph ** 2
    cond_cuadrado = cond ** 2
    rel_termo = ph / (temp + 273.15)

    # Crear la tabla con el orden exacto para XGBoost
    datos_entrada = pd.DataFrame({
        'pH': [float(ph)],
        'Conductividad': [float(cond)],
        'Temperatura': [float(temp)],
        'TDS_Estimado': [float(tds)],
        'Log_Cond': [float(log_cond)],
        'Log_Temp_Kelvin': [float(log_temp_k)],
        'pH_x_LogCond': [float(ph_x_logcond)],
        'pH_Cuadrado': [float(ph_cuadrado)],
        'Cond_Cuadrado': [float(cond_cuadrado)],
        'Relacion_Termodinamica': [float(rel_termo)]
    })
    
    # Extraemos el orden exacto de las columnas que la IA memorizo
    columnas_esperadas = ia_clasificadora.feature_names_in_
    datos_entrada = datos_entrada[columnas_esperadas]

    # Lanzar predicciones
    lsi_num = ia_regresora.predict(datos_entrada)[0]
    clase_num = ia_clasificadora.predict(datos_entrada)
    clase_texto = traductor.inverse_transform(clase_num)[0]

    # Logica de Consenso
    if lsi_num < -0.5:
        clase_reg = 'Corrosiva'
    elif lsi_num > 0.5:
        clase_reg = 'Incrustante'
    else:
        clase_reg = 'Equilibrada'

    consenso = "Aprobado" if clase_texto == clase_reg else "Revision Manual"

    # ====================================================================
    # 4. DASHBOARD DE RESULTADOS
    # ====================================================================
    st.subheader("Resultados del Analisis Quimico")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("LSI Matematico (IA 1)", f"{lsi_num:.2f}")
    col2.metric("Estado Estimado (IA 2)", clase_texto)
    col3.metric("Consenso de Seguridad", consenso)

    st.markdown("---")
    
    if consenso == "Aprobado":
        st.success(f"**VEREDICTO SEGURO:** El sistema de redundancia aprobo la prediccion. El agua presenta una tendencia **{clase_texto.upper()}**.")
    else:
        st.warning(f"**ALERTA DEL SISTEMA:** Discrepancia detectada en los modelos matematicos. La Regresion estima '{clase_reg}' mientras que la Clasificadora etiqueta como '{clase_texto}'. Se requiere intervencion manual para validacion con sensor quimico.")
