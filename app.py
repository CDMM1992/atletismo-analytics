from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st
from streamlit_option_menu import option_menu

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILOS CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Analytics Atletismo",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos personalizados para tarjetas KPI y contenedores
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. CARGA DE DATOS (Con caché para mayor velocidad)
# -----------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    # Hoja "2_Dataset_Limpio" del Excel original, exportada a CSV (carga rápida).
    # La hoja 1 del Excel es un reporte de métricas, no el dataset.
    ruta_archivo = Path(__file__).parent / "data" / "dataset_limpio.csv"
    df = pd.read_csv(ruta_archivo)
    return df, ["2_Dataset_Limpio"]


try:
    df, hojas = cargar_datos()
except FileNotFoundError:
    st.error(
        "⚠️ No se encontró `data/dataset_limpio.csv`. Verifica que la carpeta `data/` esté en el repositorio."
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Error al leer el dataset: {e}")
    st.stop()

# Identificar columnas numéricas y categóricas automáticamente
cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
# Se excluyen columnas de alta cardinalidad (ids) para no saturar los gráficos
cols_categoricas = [
    c for c in df.select_dtypes(exclude=[np.number]).columns if df[c].nunique() <= 30
]

# -----------------------------------------------------------------------------
# 3. MENÚ DE NAVEGACIÓN LATERAL (BONITO E INTERACTIVO)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=80
    )
    st.title("Atletismo Pro")
    st.markdown("---")

    selected = option_menu(
        menu_title="Navegación",
        options=[
            "Inicio & Datos",
            "Análisis Descriptivo",
            "Modelo Analítico",
            "Simulador / Predicción",
        ],
        icons=["house", "bar-chart-line", "cpu", "sliders"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {
                "padding": "5!important",
                "background-color": "#696969",
            },
            "icon": {"color": "#ff4b4b", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "2px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {
                "background-color": "#ff4b4b",
                "color": "white",
            },
        },
    )

    st.markdown("---")
    st.caption(f"📁 Fuente: {', '.join(hojas)}")
    st.caption(f"📊 Registros: {df.shape[0]} | Columnas: {df.shape[1]}")

# -----------------------------------------------------------------------------
# SECCIÓN 1: INICIO & EXPLORACIÓN DE DATOS
# -----------------------------------------------------------------------------
if selected == "Inicio & Datos":
    st.title("🏃‍♂️ Panel de Rendimiento y Analítica en Atletismo")
    st.markdown(
        "Bienvenido al sistema interactivo de exploración descriptiva y modelado predictivo."
    )

    # KPIs Generales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Registros", f"{df.shape[0]:,}")
    col2.metric("Variables Totales", df.shape[1])
    col3.metric("Variables Numéricas", len(cols_numericas))
    col4.metric("Variables Categóricas", len(cols_categoricas))

    st.markdown("### 📋 Vista Previa del Dataset Limpio")
    st.dataframe(df.head(50), use_container_width=True)

    with st.expander("🔍 Ver resumen estadístico rápido y tipos de datos"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Estadísticas Numéricas:**")
            st.dataframe(df.describe().T, use_container_width=True)
        with col_b:
            st.write("**Información de Columnas y Nulos:**")
            info_df = pd.DataFrame({
                "Tipo": df.dtypes.astype(str),
                "Nulos": df.isnull().sum(),
                "Valores Únicos": df.nunique(),
            })
            st.dataframe(info_df, use_container_width=True)

# -----------------------------------------------------------------------------
# SECCIÓN 2: ANÁLISIS DESCRIPTIVO (PLOTLY INTERACTIVO)
# -----------------------------------------------------------------------------
elif selected == "Análisis Descriptivo":
    st.title("📊 Análisis Descriptivo Exploratorio")
    st.markdown(
        "Explora distribuciones, comparativas por categoría y correlaciones entre variables."
    )

    tab1, tab2, tab3 = st.tabs([
        "📈 Distribuciones",
        "⚖️ Comparativa Categórica",
        "🔥 Mapa de Correlación",
    ])

    with tab1:
        col_ctrl, col_viz = st.columns([1, 3])
        with col_ctrl:
            var_num = st.selectbox(
                "Selecciona una variable numérica:", cols_numericas
            )
            color_cat = st.selectbox(
                "Agrupar por color (Opcional):", ["Ninguno"] + cols_categoricas
            )
            tipo_grafico = st.radio(
                "Tipo de gráfico:", ["Histograma + Boxplot", "Gráfico de Violín"]
            )

        with col_viz:
            color_param = None if color_cat == "Ninguno" else color_cat
            if tipo_grafico == "Histograma + Boxplot":
                fig = px.histogram(
                    df,
                    x=var_num,
                    color=color_param,
                    marginal="box",
                    title=f"Distribución de {var_num}",
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
            else:
                fig = px.violin(
                    df,
                    y=var_num,
                    x=color_param,
                    color=color_param,
                    box=True,
                    points="all",
                    title=f"Distribución (Violín) de {var_num}",
                    template="plotly_white",
                )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if len(cols_categoricas) > 0 and len(cols_numericas) > 0:
            c1, c2, c3 = st.columns(3)
            cat_x = c1.selectbox("Eje X (Categoría):", cols_categoricas)
            num_y = c2.selectbox("Eje Y (Métrica):", cols_numericas)
            agg_func = c3.selectbox(
                "Operación:", ["Promedio", "Mediana", "Suma", "Conteo"]
            )

            if agg_func == "Promedio":
                df_agg = df.groupby(cat_x, as_index=False)[num_y].mean()
            elif agg_func == "Mediana":
                df_agg = df.groupby(cat_x, as_index=False)[num_y].median()
            elif agg_func == "Suma":
                df_agg = df.groupby(cat_x, as_index=False)[num_y].sum()
            else:
                df_agg = df.groupby(cat_x, as_index=False)[num_y].count()

            fig_bar = px.bar(
                df_agg,
                x=cat_x,
                y=num_y,
                color=cat_x,
                text_auto=".2f",
                title=f"{agg_func} de {num_y} por {cat_x}",
                template="plotly_white",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info(
                "Se requiere al menos una columna categórica y una numérica."
            )

    with tab3:
        if len(cols_numericas) > 1:
            corr_matrix = df[cols_numericas].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Matriz de Correlación de Pearson",
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning(
                "No hay suficientes variables numéricas para calcular correlaciones."
            )

# -----------------------------------------------------------------------------
# SECCIÓN 3: MODELO ANALÍTICO / PREDICTIVO
# -----------------------------------------------------------------------------
elif selected == "Modelo Analítico":
    st.title("🧠 Modelo Analítico y Relación de Variables")
    st.markdown(
        "Configura variables predictoras y objetivo para evaluar el ajuste del modelo."
    )

    if len(cols_numericas) < 2:
        st.warning(
            "Se necesitan al menos 2 variables numéricas para construir el modelo analítico."
        )
        st.stop()

    col_config, col_res = st.columns([1, 2.5])

    with col_config:
        st.subheader("⚙️ Configuración")
        target = st.selectbox(
            "Variable Objetivo (Y):",
            cols_numericas,
            index=len(cols_numericas) - 1,
        )
        posibles_features = [c for c in cols_numericas if c != target]
        features = st.multiselect(
            "Variables Predictoras (X):",
            posibles_features,
            default=posibles_features[: min(3, len(posibles_features))],
        )
        algoritmo = st.selectbox(
            "Algoritmo:", ["Regresión Lineal", "Random Forest Regressor"]
        )
        test_size = st.slider("Tamaño de prueba (%):", 10, 40, 20) / 100

    with col_res:
        if not features:
            st.info(
                "👈 Selecciona al menos una variable predictora en el panel izquierdo."
            )
        else:
            # Preparar datos limpios para el modelo
            df_model = df[features + [target]].dropna()
            X = df_model[features]
            y = df_model[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            if algoritmo == "Regresión Lineal":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Guardar modelo en session_state para usarlo en el simulador
            st.session_state["modelo_entrenado"] = model
            st.session_state["features_modelo"] = features
            st.session_state["target_modelo"] = target

            # Métricas
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            m1, m2, m3 = st.columns(3)
            m1.metric("R² (Coef. Determinación)", f"{r2:.3f}")
            m2.metric("Error Absoluto Medio (MAE)", f"{mae:.3f}")
            m3.metric("Muestras Evaluadas", len(y_test))

            # Gráfico Real vs Predicho con Plotly
            df_eval = pd.DataFrame({"Real": y_test, "Predicho": y_pred})
            fig_pred = px.scatter(
                df_eval,
                x="Real",
                y="Predicho",
                title=f"Valores Reales vs. Predichos ({target})",
                template="plotly_white",
                opacity=0.7,
            )
            # Línea de identidad (predicción perfecta)
            min_val = min(df_eval["Real"].min(), df_eval["Predicho"].min())
            max_val = max(df_eval["Real"].max(), df_eval["Predicho"].max())
            fig_pred.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Ajuste Ideal",
                    line=dict(color="red", dash="dash"),
                )
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Importancia de variables / Coeficientes
            if algoritmo == "Random Forest Regressor":
                importancias = pd.DataFrame({
                    "Variable": features,
                    "Importancia": model.feature_importances_,
                }).sort_values(by="Importancia", ascending=True)
                fig_imp = px.bar(
                    importancias,
                    x="Importancia",
                    y="Variable",
                    orientation="h",
                    title="Importancia de Variables",
                    template="plotly_white",
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                coefs = pd.DataFrame(
                    {"Variable": features, "Coeficiente": model.coef_}
                )
                st.write("**Coeficientes de la Regresión Lineal:**")
                st.dataframe(coefs, use_container_width=True)

# -----------------------------------------------------------------------------
# SECCIÓN 4: SIMULADOR DE PREDICCIONES
# -----------------------------------------------------------------------------
elif selected == "Simulador / Predicción":
    st.title("🎛️ Simulador de Escenarios")

    if "modelo_entrenado" not in st.session_state:
        st.warning(
            "⚠️ Primero debes entrenar un modelo en la pestaña **'Modelo Analítico'** para habilitar el simulador."
        )
    else:
        model = st.session_state["modelo_entrenado"]
        features = st.session_state["features_modelo"]
        target = st.session_state["target_modelo"]

        st.markdown(
            f"Ajusta los valores de entrada para predecir **{target}**:"
        )

        inputs = {}
        cols = st.columns(min(3, len(features)))
        for i, feat in enumerate(features):
            col_actual = cols[i % len(cols)]
            min_v = float(df[feat].min())
            max_v = float(df[feat].max())
            mean_v = float(df[feat].mean())
            inputs[feat] = col_actual.slider(
                f"{feat}", min_value=min_v, max_value=max_v, value=mean_v
            )

        if st.button("🚀 Calcular Predicción", type="primary"):
            input_df = pd.DataFrame([inputs])
            resultado = model.predict(input_df)[0]
            st.success(f"### Resultado Predicho ({target}): **{resultado:.4f}**")
