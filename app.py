import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ====================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ====================================================================

st.set_page_config(
    page_title="Dashboard LSI | IoT",
    layout="wide",
    page_icon=":D"
)

st.title(":D Sistema Predictivo de Incrustaciones (LSI)")

st.write(
    "Interfaz de diagnóstico y monitoreo histórico utilizando "
    "telemetría IoT y un Ensamble de Inteligencia Artificial."
)

# ====================================================================
# 1. CARGA DE MODELOS
# ====================================================================

@st.cache_resource
def cargar_modelos():

    reg = joblib.load("ia_regresora.pkl")
    clas = joblib.load("ia_clasificadora.pkl")
    enc = joblib.load("traductor_etiquetas.pkl")

    return reg, clas, enc

try:

    ia_regresora, ia_clasificadora, traductor = cargar_modelos()

except Exception as e:

    st.error(
        ":O Error: No se encontraron los modelos .pkl "
        "en el repositorio."
    )

    st.exception(e)
    st.stop()

# ====================================================================
# 2. CONEXIÓN Y LIMPIEZA DE DATOS
# ====================================================================

URL_GOOGLE_SHEETS = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQN1fUNX4RW58QQBpT7yUivdAxivFiYpPsf9m7XNLE9ubdjSwcq6wYSuB62xSg6YwLl0r8ChcSzDjo8/"
    "pub?output=csv"
)

def convertir_numero(serie):

    """
    Convierte datos provenientes de Google Sheets a números.

    Soporta:
    - números normales
    - números almacenados como texto
    - coma decimal
    - punto decimal
    - espacios
    """

    return pd.to_numeric(
        serie
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )

@st.cache_data(ttl=10)
def obtener_datos_limpios():

    # ---------------------------------------------------------------
    # LEER GOOGLE SHEETS
    # ---------------------------------------------------------------

    df = pd.read_csv(
        URL_GOOGLE_SHEETS,
        header=0
    )

    # ---------------------------------------------------------------
    # IMPORTANTE:
    # SOLO USAREMOS A, B, C Y D
    #
    # A = tiempo
    # B = Temperatura
    # C = pH
    # D = Conductividad
    #
    # TODO LO DEMÁS SE IGNORA
    # ---------------------------------------------------------------

    if len(df.columns) < 4:

        raise ValueError(
            "Google Sheets no contiene al menos 4 columnas."
        )

    # Tomamos exclusivamente las primeras 4 columnas
    df = df.iloc[:, :4].copy()

    # Renombramos independientemente de cómo se llamen
    # originalmente en Google Sheets
    df.columns = [
        "tiempo",
        "Temperatura",
        "pH",
        "Conductividad"
    ]

    # ---------------------------------------------------------------
    # LIMPIEZA DE LAS COLUMNAS NUMÉRICAS
    # ---------------------------------------------------------------

    df["Temperatura"] = convertir_numero(
        df["Temperatura"]
    )

    df["pH"] = convertir_numero(
        df["pH"]
    )

    df["Conductividad"] = convertir_numero(
        df["Conductividad"]
    )

    # ---------------------------------------------------------------
    # LIMPIEZA DE FECHA
    # ---------------------------------------------------------------

    df["tiempo"] = pd.to_datetime(
        df["tiempo"],
        errors="coerce",
        dayfirst=True
    )

    # ---------------------------------------------------------------
    # ELIMINAR FILAS INVALIDAS
    # ---------------------------------------------------------------

    df = df.dropna(
        subset=[
            "tiempo",
            "Temperatura",
            "pH",
            "Conductividad"
        ]
    )

    # ---------------------------------------------------------------
    # ORDEN CRONOLÓGICO
    # ---------------------------------------------------------------

    df = df.sort_values(
        by="tiempo"
    )

    df = df.reset_index(
        drop=True
    )

    return df

# ====================================================================
# 3. SELECCIÓN DEL MODO DE CONEXIÓN
# ====================================================================

st.sidebar.header("Origen de Datos")

modo_conexion = st.sidebar.radio(
    "Selecciona el modo:",
    [
        "Telemetria (Google Sheets)",
        "Simulacion Manual"
    ]
)

# ====================================================================
# 4. OBTENER DATOS
# ====================================================================

if modo_conexion == "Telemetria (Google Sheets)":

    try:

        df_telemetria = obtener_datos_limpios()

        # -----------------------------------------------------------
        # ÚLTIMA LECTURA
        # -----------------------------------------------------------

        ultima_lectura = df_telemetria.iloc[-1]

        fecha_lectura = ultima_lectura["tiempo"]

        temp = float(
            ultima_lectura["Temperatura"]
        )

        ph = float(
            ultima_lectura["pH"]
        )

        cond = float(
            ultima_lectura["Conductividad"]
        )

        # -----------------------------------------------------------
        # INFORMACIÓN EN SIDEBAR
        # -----------------------------------------------------------

        st.sidebar.success(
            ":D Conectado a la base de datos "
            "hecha y administrada por Santiago"
        )

        st.sidebar.markdown(
            f"**Última lectura:** {fecha_lectura}"
        )

        st.sidebar.metric(
            "pH",
            f"{ph:.2f}"
        )

        st.sidebar.metric(
            "Conductividad",
            f"{cond:.2f} µS/cm"
        )

        st.sidebar.metric(
            "Temperatura",
            f"{temp:.2f} °C"
        )

    except Exception as e:

        st.sidebar.error(
            ":O Error al procesar los datos."
        )

        st.error(
            "El formato de las columnas A, B, C y D "
            "no es compatible con el sistema."
        )

        st.exception(e)

        st.stop()

# ====================================================================
# 5. SIMULACIÓN MANUAL
# ====================================================================

else:

    st.sidebar.subheader(
        "Lecturas Manuales"
    )

    ph = st.sidebar.number_input(
        "pH del Agua",
        min_value=0.0,
        max_value=14.0,
        value=7.2,
        step=0.1
    )

    cond = st.sidebar.number_input(
        "Conductividad (µS/cm)",
        min_value=0.0,
        max_value=5000.0,
        value=800.0,
        step=10.0
    )

    temp = st.sidebar.number_input(
        "Temperatura (°C)",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=0.1
    )

# ====================================================================
# 6. CONVERSIÓN FINAL A FLOAT
# ====================================================================

ph = float(ph)
cond = float(cond)
temp = float(temp)

# ====================================================================
# 7. PROCESAMIENTO E INTELIGENCIA ARTIFICIAL
# ====================================================================

# ---------------------------------------------------------------
# VARIABLES BASE
# ---------------------------------------------------------------

tds = cond * 0.5

log_cond = np.log10(
    cond + 1
)

log_temp_k = np.log10(
    temp + 273.15
)

ph_x_logcond = (
    ph * log_cond
)

ph_cuadrado = (
    ph ** 2
)

cond_cuadrado = (
    cond ** 2
)

rel_termo = (
    ph / (temp + 273.15)
)

# ====================================================================
# 8. DATAFRAME DE ENTRADA PARA LOS MODELOS
# ====================================================================

datos_entrada = pd.DataFrame({

    "pH": [
        ph
    ],

    "Conductividad": [
        cond
    ],

    "Temperatura": [
        temp
    ],

    "TDS_Estimado": [
        tds
    ],

    "Log_Cond": [
        log_cond
    ],

    "Log_Temp_Kelvin": [
        log_temp_k
    ],

    "pH_x_LogCond": [
        ph_x_logcond
    ],

    "pH_Cuadrado": [
        ph_cuadrado
    ],

    "Cond_Cuadrado": [
        cond_cuadrado
    ],

    "Relacion_Termodinamica": [
        rel_termo
    ]
})

# ====================================================================
# 9. ORDEN EXACTO DE LAS VARIABLES DEL MODELO
# ====================================================================

try:

    columnas_esperadas = (
        ia_clasificadora.feature_names_in_
    )

except AttributeError:

    columnas_esperadas = datos_entrada.columns

# Verificar que todas las características existan

faltantes = [
    columna
    for columna in columnas_esperadas
    if columna not in datos_entrada.columns
]

if faltantes:

    st.error(
        ":O El modelo requiere columnas que "
        "no fueron generadas: "
        f"{faltantes}"
    )

    st.stop()

datos_entrada = datos_entrada[
    columnas_esperadas
]

# ====================================================================
# 10. PREDICCIONES
# ====================================================================

try:

    # ---------------------------------------------------------------
    # MODELO REGRESOR
    # ---------------------------------------------------------------

    lsi_num = (
        ia_regresora
        .predict(datos_entrada)[0]
    )

    # ---------------------------------------------------------------
    # MODELO CLASIFICADOR
    # ---------------------------------------------------------------

    clase_num = (
        ia_clasificadora
        .predict(datos_entrada)
    )

    clase_texto = (
        traductor
        .inverse_transform(clase_num)[0]
    )

except Exception as e:

    st.error(
        ":O Error al ejecutar los modelos de IA."
    )

    st.exception(e)

    st.stop()

# ====================================================================
# 11. CLASIFICACIÓN DEL MODELO REGRESOR
# ====================================================================

if lsi_num < -0.5:

    clase_reg = "Corrosiva"

elif lsi_num > 0.5:

    clase_reg = "Incrustante"

else:

    clase_reg = "Equilibrada"

# ====================================================================
# 12. LÓGICA DE CONSENSO
# ====================================================================

if clase_texto == clase_reg:

    consenso = ":D Aprobado"

else:

    consenso = (
        ":O Alerta, se requiere de revision manual"
    )

# ====================================================================
# 13. DASHBOARD DE RESULTADOS
# ====================================================================

st.subheader(
    "Diagnóstico Quimico en Tiempo Real"
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "LSI Matematico (IA 1)",
    f"{lsi_num:.2f}"
)

col2.metric(
    "Estado Estimado (IA 2)",
    clase_texto
)

col3.metric(
    "Verificacion de seguridad",
    consenso
)

# ====================================================================
# 14. VEREDICTO
# ====================================================================

if consenso == ":D Aprobado":

    st.success(
        f"**VEREDICTO SEGURO:** "
        f"El sistema de redundancia aprobó la prediccion, "
        f"por lo que es probable que tenga razon. "
        f"El agua presenta tendencia "
        f"**{clase_texto.upper()}**."
    )

else:

    st.warning(
        f"**ALERTA DEL SISTEMA:** "
        f"Discrepancia matematica. "
        f"Regresión estima '{clase_reg}' "
        f"mientras que Clasificadora etiqueta "
        f"'{clase_texto}'."
    )

# ====================================================================
# 16. DASHBOARD HISTÓRICO
# ====================================================================

if modo_conexion == "Telemetria (Google Sheets)":

    st.markdown("---")

    st.subheader(
        "Análisis historico - últimos 7 días de actividad"
    )

    # ---------------------------------------------------------------
    # CREAR FECHA PURA
    # ---------------------------------------------------------------

    df_telemetria["Fecha_Pura"] = (
        df_telemetria["tiempo"].dt.date
    )

    # ---------------------------------------------------------------
    # OBTENER LOS DÍAS REGISTRADOS
    # ---------------------------------------------------------------

    dias_unicos_registrados = sorted(
        df_telemetria["Fecha_Pura"]
        .dropna()
        .unique()
    )

    # ---------------------------------------------------------------
    # ÚLTIMOS 7 DÍAS DISPONIBLES
    # ---------------------------------------------------------------

    ultimos_7_dias = (
        dias_unicos_registrados[-7:]
    )

    # ---------------------------------------------------------------
    # FILTRAR
    # ---------------------------------------------------------------

    df_plot = df_telemetria[
        df_telemetria["Fecha_Pura"].isin(
            ultimos_7_dias
        )
    ].copy()

    # =================================================================
    # 17. CREAR CARACTERÍSTICAS HISTÓRICAS
    # =================================================================

    df_plot_features = pd.DataFrame({

        "pH":
            df_plot["pH"].astype(float),

        "Conductividad":
            df_plot["Conductividad"].astype(float),

        "Temperatura":
            df_plot["Temperatura"].astype(float)

    })

    # ---------------------------------------------------------------
    # VARIABLES ESTIMADAS
    # ---------------------------------------------------------------

    df_plot_features["TDS_Estimado"] = (
        df_plot_features["Conductividad"]
        * 0.5
    )

    df_plot_features["Log_Cond"] = np.log10(
        df_plot_features["Conductividad"] + 1
    )

    df_plot_features["Log_Temp_Kelvin"] = np.log10(
        df_plot_features["Temperatura"] + 273.15
    )

    df_plot_features["pH_x_LogCond"] = (
        df_plot_features["pH"]
        *
        df_plot_features["Log_Cond"]
    )

    df_plot_features["pH_Cuadrado"] = (
        df_plot_features["pH"] ** 2
    )

    df_plot_features["Cond_Cuadrado"] = (
        df_plot_features["Conductividad"] ** 2
    )

    df_plot_features["Relacion_Termodinamica"] = (
        df_plot_features["pH"]
        /
        (
            df_plot_features["Temperatura"]
            + 273.15
        )
    )

    # ---------------------------------------------------------------
    # ORDEN DE COLUMNAS DEL MODELO
    # ---------------------------------------------------------------

    df_plot_features = (
        df_plot_features[
            columnas_esperadas
        ]
    )

    # =================================================================
    # 18. PREDICCIÓN HISTÓRICA
    # =================================================================

    try:

        df_plot["Curva LSI Predicha"] = (
            ia_regresora.predict(
                df_plot_features
            )
        )

    except Exception as e:

        st.error(
            ":O Error al generar la predicción histórica."
        )

        st.exception(e)

        st.stop()

    # =================================================================
    # 19. USAR TIEMPO COMO ÍNDICE
    # =================================================================

    df_plot = df_plot.set_index(
        "tiempo"
    )

    # =================================================================
    # 20. PESTAÑAS
    # =================================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "Curva de Corrosión/Sarro (LSI)",
            "pH y Temperatura",
            "Sólidos (Conductividad)"
        ]
    )

    # ================================================================
    # TAB 1
    # ================================================================

    with tab1:

        st.write(
            "Variación de la tendencia incrustante "
            "calculada por IA a lo largo del registro:"
        )

        st.line_chart(
            df_plot[
                "Curva LSI Predicha"
            ],
            height=350
        )

    # ================================================================
    # TAB 2
    # ================================================================

    with tab2:

        st.write(
            "Comportamiento físico-químico base:"
        )

        st.line_chart(
            df_plot[
                [
                    "pH",
                    "Temperatura"
                ]
            ],
            height=350
        )

    # ================================================================
    # TAB 3
    # ================================================================

    with tab3:

        st.write(
            "Nivel de mineralización:"
        )

        st.line_chart(
            df_plot[
                "Conductividad"
            ],
            height=350
        )

# ====================================================================
# 21. PIE DE PÁGINA
# ====================================================================

st.markdown("---")

st.caption(
    ":D Sistema de monitoreo y predicción LSI "
    "mediante telemetría IoT e Inteligencia Artificial."
)
