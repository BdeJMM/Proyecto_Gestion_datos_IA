import streamlit as st
import pandas as pd
from conexion import get_engine

st.set_page_config(page_title="Pipeline Dashboard", layout="wide")
st.title("Dashboard — Gestión Datos IA")

engine = get_engine()

# Métricas del modelo
st.header("Métricas del modelo")
df_metricas = pd.read_sql("""
    SELECT modelo, auc_roc, f1_score, accuracy, precision_m, recall, fecha_carga
    FROM metricas_modelo
    ORDER BY fecha_carga DESC
""", engine)
st.dataframe(df_metricas)

col1, col2, col3 = st.columns(3)
if not df_metricas.empty:
    ultimo = df_metricas.iloc[0]
    col1.metric("AUC-ROC", f"{ultimo['auc_roc']:.4f}")
    col2.metric("F1-Score", f"{ultimo['f1_score']:.4f}")
    col3.metric("Accuracy", f"{ultimo['accuracy']:.4f}")

# Rendimiento del pipeline
st.header("Rendimiento del pipeline")
df_rend = pd.read_sql("""
    SELECT etapa, latencia_seg, filas_entrada, filas_salida, completitud, exito
    FROM rendimiento_pipeline
    ORDER BY fecha_carga DESC
    FETCH FIRST 20 ROWS ONLY
""", engine)
st.dataframe(df_rend)
st.bar_chart(df_rend.set_index("etapa")["latencia_seg"])