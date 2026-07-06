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

    # ── Único selector — la condición del pipeline ────────────────────────────
    condiciones_disponibles = df_ejecucion["condicion"].unique().tolist()
    condicion_sel = st.selectbox(
        "Condición del pipeline",
        condiciones_disponibles,
        format_func=lambda c: ETIQUETAS_CONDICION.get(c, c),
    )

    df_cond = df_ejecucion[df_ejecucion["condicion"] == condicion_sel]
    modelos = df_cond["modelo"].unique().tolist()  # ej. ["Regresión Logística", "XGBoost"]

    # ── KPIs — Regresión Logística y XGBoost lado a lado ─────────────────────
    st.header("Métricas del modelo")
    cols_kpi = st.columns(len(modelos))
    for col, modelo in zip(cols_kpi, modelos):
        fila = df_cond[df_cond["modelo"] == modelo].iloc[0]
        with col:
            st.subheader(modelo)
            st.metric("AUC-ROC", f"{fila['auc_roc']:.4f}")
            st.metric("Gini", f"{fila['gini']:.4f}")
            st.metric("F1-Score", f"{fila['f1_score']:.4f}")
            st.metric("Accuracy", f"{fila['accuracy']:.4f}")

    # ── Matriz de confusión — ambos modelos lado a lado ───────────────────────
    st.header("Matriz de confusión")
    df_matriz = pd.read_sql(
        "SELECT modelo, tn, fp, fn, tp FROM matriz_confusion "
        "WHERE ejecucion = %(ejecucion)s AND condicion = %(condicion)s",
        engine, params={"ejecucion": ejecucion_reciente, "condicion": condicion_sel},
    )
    if not df_matriz.empty:
        cols_matriz = st.columns(len(modelos))
        for col, modelo in zip(cols_matriz, modelos):
            fila_m = df_matriz[df_matriz["modelo"] == modelo]
            if fila_m.empty:
                continue
            fila = fila_m.iloc[0]
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
            ax.set_title(modelo)
            with col:
                st.pyplot(fig)
    else:
        st.info("Todavía no hay datos de matriz de confusión para esta selección.")

    # ── Curva ROC — ambos modelos superpuestos en un mismo gráfico ───────────
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

    # ── Rendimiento del pipeline — misma condición elegida en el selector ────
    st.header("Métricas de rendimiento del pipeline")
    df_rend = pd.read_sql(
        "SELECT etapa, latencia_seg, filas_entrada, filas_salida, completitud, exito "
        "FROM rendimiento_pipeline "
        "WHERE ejecucion = %(ejecucion)s AND condicion = %(condicion)s "
        "ORDER BY fecha_carga",
        engine, params={"ejecucion": ejecucion_reciente, "condicion": condicion_sel},
    )

    if not df_rend.empty:
        st.subheader("Tiempo (segundos) que tardó cada etapa")
        st.bar_chart(df_rend.set_index("etapa")["latencia_seg"])
    else:
        st.info("Todavía no hay datos de rendimiento del pipeline para esta selección.")
