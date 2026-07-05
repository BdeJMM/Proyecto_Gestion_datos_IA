import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from conexion import get_engine

st.set_page_config(page_title="Pipeline Dashboard", layout="wide")
st.title("Dashboard — Gestión Datos IA")

engine = get_engine()

df_metricas = pd.read_sql("""
    SELECT ejecucion, modelo, auc_roc, gini, f1_score, accuracy, precision_m, recall, fecha_carga
    FROM metricas_modelo
    ORDER BY fecha_carga DESC
""", engine)

if not df_metricas.empty:
    ultimo = df_metricas.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", f"{ultimo['auc_roc']:.4f}")
    col2.metric("Gini", f"{ultimo['gini']:.4f}")
    col3.metric("F1-Score", f"{ultimo['f1_score']:.4f}")
    col4.metric("Accuracy", f"{ultimo['accuracy']:.4f}")

else:
    st.info("Todavía no hay métricas del modelo cargadas.")

st.header("Matriz de confusión")

if not df_metricas.empty:
    ejecucion_reciente = ultimo["ejecucion"]
    df_matriz = pd.read_sql(
        "SELECT modelo, tn, fp, fn, tp FROM matriz_confusion WHERE ejecucion = %(ejecucion)s",
        engine, params={"ejecucion": ejecucion_reciente},
    )

    if not df_matriz.empty:
        modelos_disponibles = df_matriz["modelo"].tolist()
        modelo_matriz = st.selectbox("Modelo — matriz de confusión", modelos_disponibles)
        fila = df_matriz[df_matriz["modelo"] == modelo_matriz].iloc[0]

        matriz = [[fila["tn"], fila["fp"]], [fila["fn"], fila["tp"]]]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(matriz, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(matriz[i][j]), ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1], labels=["no", "yes"])
        ax.set_yticks([0, 1], labels=["no", "yes"])
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de confusión — {modelo_matriz}")
        st.pyplot(fig)
    else:
        st.info("Todavía no hay datos de matriz de confusión para esta ejecución.")
else:
    st.info("Todavía no hay métricas del modelo cargadas.")

# ── Curva ROC ──────────────────────────────────────────────────────────────
st.header("Curva ROC")

if not df_metricas.empty:
    df_roc = pd.read_sql(
        "SELECT modelo, fpr, tpr, auc_roc FROM curva_roc WHERE ejecucion = %(ejecucion)s",
        engine, params={"ejecucion": ejecucion_reciente},
    )

    if not df_roc.empty:
        fig, ax = plt.subplots(figsize=(5, 5))
        for _, fila in df_roc.iterrows():
            fpr = json.loads(fila["fpr"])
            tpr = json.loads(fila["tpr"])
            ax.plot(fpr, tpr, label=f"{fila['modelo']} (AUC={fila['auc_roc']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatorio")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title("Curva ROC — comparación de modelos")
        ax.legend(loc="lower right")
        st.pyplot(fig)
    else:
        st.info("Todavía no hay datos de curva ROC para esta ejecución.")
else:
    st.info("Todavía no hay métricas del modelo cargadas.")


st.header("Métricas de rendimiento del pipeline")

df_rend = pd.read_sql("""
    SELECT etapa, latencia_seg, filas_entrada, filas_salida, completitud, exito
    FROM rendimiento_pipeline
    ORDER BY fecha_carga DESC
    LIMIT 20
""", engine)

if not df_rend.empty:
    st.subheader("Tiempo (segundos) que tardó cada etapa")
    st.bar_chart(df_rend.set_index("etapa")["latencia_seg"])
else:
    st.info("Todavía no hay datos de rendimiento del pipeline.")