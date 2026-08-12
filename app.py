import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página (Más ancha para gráficos)
st.set_page_config(page_title="Dashboard LSI | IoT", layout="wide", page_icon=":D")

st.title(":D Sistema Predictivo de Incrustaciones (LSI)")
st.write("Interfaz de diagnóstico y monitoreo histórico utilizando telemetría IoT y un Ensamble de Inteligencia Artificial.")

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
# 2. CONEXIÓN Y LIMPIEZA DE DATOS (IoT en Tiempo Real)
# ====================================================================
URL_GOOGLE_SHEETS = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQN1fUNX4RW58QQBpT7yUivdAxivFiYpPsf9m7XNLE9ubdjSwcq6wYSuB62xSg6YwLl0r8ChcSzDjo8/pub?output=csv"

@st.cache_data(ttl=10) # Se refresca cada 10 segundos
def obtener_datos_limpios():
    # Leemos el archivo crudo
    df = pd.read_csv(URL_GOOGLE_SHEETS)
    
    # LIMPIEZA DE FORMATO: Reemplazar comas por puntos en Temp, pH y Cond
    # Columnas esperadas: 0(Tiempo), 1(Temp), 2(pH), 3(Cond)
    for i in [1, 2, 3]:
        df.iloc[:, i] = df.iloc[:, i].astype(str).str.replace(',', '.').astype(float)
    
    # Arreglar la fecha para que Python la entienda cronológicamente
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[df.columns[0]]) # Elimina filas si la fecha falló
    df = df.sort_values(by=df.columns[0])  # Ordena de más viejo a más nuevo
    
    return df

st.sidebar.header("Origen de Datos")
modo_conexion = st.sidebar.radio("Selecciona el modo:", ["Telemetria (Google Sheets)", "Simulacion Manual"])

if modo_conexion == "Telemetria (Google Sheets)":
    try:
        df_telemetria = obtener_datos_limpios()
        
        # Extraemos la última fila (la más reciente)
        ultima_lectura = df_telemetria.iloc[-1]
        
        fecha_lectura = str(ultima_lectura.iloc[0])
        temp = float(ultima_lectura.iloc[1])
        ph = float(ultima_lectura.iloc[2])
        cond = float(ultima_lectura.iloc[3])
        
        st.sidebar.success(":D Conectado a la base de datos hecha y administrada por santiago")
        st.sidebar.markdown(f"**Última lectura:** {fecha_lectura}")
        st.sidebar.metric("pH", f"{ph}")
        st.sidebar.metric("Conductividad", f"{cond} µS/cm")
        st.sidebar.metric("Temperatura", f"{temp} °C")
        
    except Exception as e:
        st.sidebar.error(f":O Error al procesar los datos. El formato de origen no es compatible.")
        st.stop()
else:
    st.sidebar.subheader("Lecturas Manuales")
    ph = st.sidebar.number_input("pH del Agua", min_value=0.0, max_value=14.0, value=7.2, step=0.1)
    cond = st.sidebar.number_input("Conductividad (µS/cm)", min_value=0.0, max_value=5000.0, value=800.0, step=10.0)
    temp = st.sidebar.number_input("Temperatura (°C)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)

# ====================================================================
# 3. PROCESAMIENTO E INTELIGENCIA ARTIFICIAL
# ====================================================================
# Ingeniería de Características de la lectura actual
tds = cond * 0.5
log_cond = np.log10(cond + 1)
log_temp_k = np.log10(temp + 273.15)
ph_x_logcond = ph * log_cond
ph_cuadrado = ph ** 2
cond_cuadrado = cond ** 2
rel_termo = ph / (temp + 273.15)

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

# Alinear columnas con la memoria de la IA
columnas_esperadas = ia_clasificadora.feature_names_in_
datos_entrada = datos_entrada[columnas_esperadas]

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

consenso = ":D Aprobado" if clase_texto == clase_reg else ":O Alerta, se requiere de revision manual"

# ====================================================================
# 4. DASHBOARD DE RESULTADOS (TIEMPO REAL)
# ====================================================================
st.subheader(" Diagnóstico Quimico en Tiempo Real")
col1, col2, col3 = st.columns(3)

col1.metric("LSI Matematico (IA 1)", f"{lsi_num:.2f}")
col2.metric("Estado Estimado (IA 2)", clase_texto)
col3.metric("Verificacion de seguridad", consenso)

if consenso == ":D Aprobado":
    st.success(f"**VEREDICTO SEGURO:** El sistema de redundancia aprobó la prediccion, por lo que es probable que tenga razon. El agua presenta tendencia **{clase_texto.upper()}**.")
else:
    st.warning(f"**ALERTA DEL SISTEMA:** Discrepancia matematica. Regresión estima '{clase_reg}' mientras que Clasificadora etiqueta '{clase_texto}'.")

st.markdown("---")

# ====================================================================
# 5. DASHBOARD HISTÓRICO (ÚLTIMOS 7 DÍAS REGISTRADOS)
# ====================================================================
if modo_conexion == "Telemetria (Google Sheets)":
    st.subheader("Análisis historico ultimos, 7 días de actividad)")
    
    # Extraer la fecha ignorando la hora
    df_telemetria['Fecha_Pura'] = df_telemetria.iloc[:, 0].dt.date
    
    # Identificar los últimos 7 días únicos registrados
    dias_unicos_registrados = sorted(df_telemetria['Fecha_Pura'].unique())
    ultimos_7_dias = dias_unicos_registrados[-7:] # Corta la lista a máximo 7 elementos
    
    # Filtrar la tabla para mostrar solo esos días
    df_plot = df_telemetria[df_telemetria['Fecha_Pura'].isin(ultimos_7_dias)].copy()
    df_plot.set_index(df_plot.columns[0], inplace=True) # Poner el tiempo como eje X
    
    # ---- CALCULAR EL LSI PARA TODO EL HISTORIAL ----
    df_plot_features = pd.DataFrame({
        'pH': df_plot.iloc[:, 2],
        'Conductividad': df_plot.iloc[:, 3],
        'Temperatura': df_plot.iloc[:, 1],
    })
    
    df_plot_features['TDS_Estimado'] = df_plot_features['Conductividad'] * 0.5
    df_plot_features['Log_Cond'] = np.log10(df_plot_features['Conductividad'] + 1)
    df_plot_features['Log_Temp_Kelvin'] = np.log10(df_plot_features['Temperatura'] + 273.15)
    df_plot_features['pH_x_LogCond'] = df_plot_features['pH'] * df_plot_features['Log_Cond']
    df_plot_features['pH_Cuadrado'] = df_plot_features['pH'] ** 2
    df_plot_features['Cond_Cuadrado'] = df_plot_features['Conductividad'] ** 2
    df_plot_features['Relacion_Termodinamica'] = df_plot_features['pH'] / (df_plot_features['Temperatura'] + 273.15)
    
    df_plot_features = df_plot_features[columnas_esperadas]
    
    # Inyectar la curva predicha por la IA a lo largo del tiempo
    df_plot['Curva LSI Predicha'] = ia_regresora.predict(df_plot_features)
    
    # Mostrar Gráficas en Pestañas
    tab1, tab2, tab3 = st.tabs(["Curva de Corrosión/Sarro (LSI)", "pH y Temperatura", "Sólidos (Conductividad)"])
    
    with tab1:
        st.write("Variación de la tendencia incrustante a lo largo de la semana:")
        st.line_chart(df_plot['Curva LSI Predicha'], color="#FF4B4B")
    
    with tab2:
        st.write("Comportamiento físico-químico base:")
        # Grafica la Temperatura (col 1) y el pH (col 2)
        st.line_chart(df_plot.iloc[:, [1, 2]], height=350)
        
    with tab3:
        st.write("Nivel de mineralización:")
        # Grafica la Conductividad (col 3)
        st.line_chart(df_plot.iloc[:, 3], color="#0068C9")
