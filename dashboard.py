import streamlit as st
import pandas as pd
from conexion import get_engine

st.set_page_config(page_title="Pipeline Dashboard", layout="wide")
st.title("Dashboard — Gestión Datos IA")

engine = get_engine()

# ── Métricas del modelo ───────────────────────────────────────────────────────
st.header("Métricas del modelo")
df_metricas = pd.read_sql("""
    SELECT modelo, auc_roc, f1_score, accuracy, precision_m, recall, fecha_carga
    FROM metricas_modelo
    ORDER BY fecha_carga DESC
""", engine)
st.dataframe(df_metricas, use_container_width=True)

if not df_metricas.empty:
    ultimo = df_metricas.iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC",  f"{ultimo['auc_roc']:.4f}")
    col2.metric("F1-Score",  f"{ultimo['f1_score']:.4f}")
    col3.metric("Accuracy",  f"{ultimo['accuracy']:.4f}")

# ── Evolución AUC por ejecución ───────────────────────────────────────────────
st.header("Evolución AUC por ejecución")
if not df_metricas.empty:
    df_auc = df_metricas[["modelo", "auc_roc", "fecha_carga"]].copy()
    st.line_chart(df_auc.set_index("fecha_carga")["auc_roc"])

# ── Rendimiento del pipeline ──────────────────────────────────────────────────
st.header("Rendimiento del pipeline")
df_rend = pd.read_sql("""
    SELECT etapa, latencia_seg, filas_entrada, filas_salida, completitud, exito
    FROM rendimiento_pipeline
    ORDER BY fecha_carga DESC
    LIMIT 20
""", engine)
st.dataframe(df_rend, use_container_width=True)

if not df_rend.empty:
    st.bar_chart(df_rend.set_index("etapa")["latencia_seg"])

# ── Depósitos por trabajo ─────────────────────────────────────────────────────
st.header("Depósitos por tipo de trabajo")
df_trabajo = pd.read_sql("""
    SELECT job,
           COUNT(*) AS total,
           SUM(CASE WHEN deposit = 'yes' THEN 1 ELSE 0 END) AS depositaron,
           ROUND(
               SUM(CASE WHEN deposit = 'yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
           ) AS pct_deposito
    FROM clientes_validos
    GROUP BY job
    ORDER BY total DESC
""", engine)
st.dataframe(df_trabajo, use_container_width=True)
if not df_trabajo.empty:
    st.bar_chart(df_trabajo.set_index("job")["pct_deposito"])

# ── Últimos errores ───────────────────────────────────────────────────────────
st.header("Últimos registros erróneos")
df_errores = pd.read_sql("""
    SELECT ejecucion, condicion, fila_original, motivo_error, age, job, deposit, fecha_carga
    FROM registros_erroneos
    ORDER BY fecha_carga DESC
    LIMIT 20
""", engine)
st.dataframe(df_errores, use_container_width=True)

# ── Resumen de ejecuciones ────────────────────────────────────────────────────
st.header("Resumen de ejecuciones")
df_resumen = pd.read_sql("""
    SELECT v.ejecucion, v.condicion, v.total_validos,
           COALESCE(e.total_errores, 0) AS total_errores, v.fecha_carga
    FROM (
        SELECT ejecucion, condicion, COUNT(*) AS total_validos, MAX(fecha_carga) AS fecha_carga
        FROM clientes_validos
        GROUP BY ejecucion, condicion
    ) v
    LEFT JOIN (
        SELECT ejecucion, condicion, COUNT(*) AS total_errores
        FROM registros_erroneos
        GROUP BY ejecucion, condicion
    ) e ON v.ejecucion = e.ejecucion AND v.condicion = e.condicion
    ORDER BY v.fecha_carga DESC
""", engine)
st.dataframe(df_resumen, use_container_width=True)
