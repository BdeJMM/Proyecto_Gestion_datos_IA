import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from conexion import get_engine

st.set_page_config(page_title="Pipeline Dashboard", layout="wide")
st.title("Dashboard — Gestión Datos IA")

engine = get_engine()

ETIQUETAS_CONDICION = {
    "completo": "Dataset completo",
    "50pct": "Dataset al 50%",
}

df_metricas = pd.read_sql("""
    SELECT ejecucion, condicion, modelo, auc_roc, gini, f1_score, accuracy, precision_m, recall, fecha_carga
    FROM metricas_modelo
    ORDER BY fecha_carga DESC
""", engine)

if df_metricas.empty:
    st.info("Todavía no hay métricas del modelo cargadas.")
else:
    ejecucion_reciente = df_metricas.iloc[0]["ejecucion"]
    df_ejecucion = df_metricas[df_metricas["ejecucion"] == ejecucion_reciente]

    # ── Selectores compartidos — arriba de todo, controlan toda la página ────
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        condiciones_disponibles = df_ejecucion["condicion"].unique().tolist()
        condicion_sel = st.selectbox(
            "Condición del pipeline",
            condiciones_disponibles,
            format_func=lambda c: ETIQUETAS_CONDICION.get(c, c),
        )
    with col_sel2:
        modelos_disponibles = (
            df_ejecucion[df_ejecucion["condicion"] == condicion_sel]["modelo"]
            .unique().tolist()
        )
        modelo_sel = st.selectbox("Modelo", modelos_disponibles)

    fila_sel = df_ejecucion[
        (df_ejecucion["condicion"] == condicion_sel)
        & (df_ejecucion["modelo"] == modelo_sel)
    ].iloc[0]

    # ── KPIs — ahora sí reaccionan a los selectores de arriba ────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", f"{fila_sel['auc_roc']:.4f}")
    col2.metric("Gini", f"{fila_sel['gini']:.4f}")
    col3.metric("F1-Score", f"{fila_sel['f1_score']:.4f}")
    col4.metric("Accuracy", f"{fila_sel['accuracy']:.4f}")

    # ── Matriz de confusión ───────────────────────────────────────────────────
    st.header("Matriz de confusión")
    df_matriz = pd.read_sql(
        "SELECT modelo, tn, fp, fn, tp FROM matriz_confusion "
        "WHERE ejecucion = %(ejecucion)s AND condicion = %(condicion)s",
        engine, params={"ejecucion": ejecucion_reciente, "condicion": condicion_sel},
    )
    fila_matriz = df_matriz[df_matriz["modelo"] == modelo_sel]
    if not fila_matriz.empty:
        fila = fila_matriz.iloc[0]
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
        ax.set_title(f"Matriz de confusión — {modelo_sel}")
        st.pyplot(fig)
    else:
        st.info("Todavía no hay datos de matriz de confusión para esta selección.")

    # ── Curva ROC — compara ambos modelos dentro de la condición elegida ─────
    st.header("Curva ROC")
    df_roc = pd.read_sql(
        "SELECT modelo, fpr, tpr, auc_roc FROM curva_roc "
        "WHERE ejecucion = %(ejecucion)s AND condicion = %(condicion)s",
        engine, params={"ejecucion": ejecucion_reciente, "condicion": condicion_sel},
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
        etiqueta_cond = ETIQUETAS_CONDICION.get(condicion_sel, condicion_sel)
        ax.set_title(f"Curva ROC — {etiqueta_cond}")
        ax.legend(loc="lower right")
        st.pyplot(fig)
    else:
        st.info("Todavía no hay datos de curva ROC para esta selección.")

# ── Rendimiento del pipeline (no depende de los selectores de arriba) ────────
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
