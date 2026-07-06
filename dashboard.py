import streamlit as st
import pandas as pd
import json
import subprocess
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
from conexion import get_engine

st.set_page_config(page_title="Pipeline Dashboard", layout="wide")
st.title("Dashboard — Gestión Datos IA")

engine = get_engine()

ETIQUETAS_CONDICION = {
    "completo": "Dataset completo",
    "50pct": "Dataset al 50%",
}

# ── Ruta al script del pipeline (misma carpeta que este dashboard) ───────────
RUTA_PIPELINE = Path(__file__).resolve().parent / "pipeline.py"


def _ejecutar_pipeline_subproceso(log_placeholder) -> int:
    """Lanza pipeline.py como subproceso, va mostrando su log en vivo y lo
    guarda en session_state para que quede disponible incluso después de
    terminar (y se pueda minimizar sin que desaparezca)."""
    proceso = subprocess.Popen(
        [sys.executable, str(RUTA_PIPELINE)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(RUTA_PIPELINE.parent),
    )

    lineas = []
    for linea in proceso.stdout:
        lineas.append(linea)
        texto_actual = "".join(lineas[-300:])
        st.session_state.pipeline_log = texto_actual
        log_placeholder.code(texto_actual, language="text")

    proceso.wait()
    return proceso.returncode


# ── Panel lateral — ejecutar el pipeline desde la UI ─────────────────────────
st.sidebar.header("⚙️ Pipeline")

if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "pipeline_log" not in st.session_state:
    st.session_state.pipeline_log = ""
if "pipeline_exit_code" not in st.session_state:
    st.session_state.pipeline_exit_code = None

lanzar = st.sidebar.button(
    "▶️ Ejecutar pipeline completo",
    disabled=st.session_state.pipeline_running,
    use_container_width=True,
)

if lanzar and not st.session_state.pipeline_running:
    st.session_state.pipeline_running = True
    st.session_state.pipeline_exit_code = None
    st.rerun()

if st.session_state.pipeline_running:
    st.sidebar.info("Ejecutando pipeline… esto puede tardar unos minutos.")

# El expander del log SIEMPRE se muestra si hay algo que mostrar (corriendo
# o ya terminado), así se puede minimizar/expandir sin perder el contenido.
if st.session_state.pipeline_running or st.session_state.pipeline_log:
    with st.expander("📜 Log del pipeline", expanded=st.session_state.pipeline_running):
        log_placeholder = st.empty()
        log_placeholder.code(st.session_state.pipeline_log, language="text")

        if st.session_state.pipeline_running:
            codigo_salida = _ejecutar_pipeline_subproceso(log_placeholder)
            st.session_state.pipeline_running = False
            st.session_state.pipeline_exit_code = codigo_salida
            # Recargar la app para que las métricas de abajo (df_metricas, etc.)
            # se actualicen con los datos nuevos. El log ya quedó guardado en
            # session_state, así que no desaparece con este rerun.
            time.sleep(1)
            st.rerun()

if st.session_state.pipeline_exit_code is not None and not st.session_state.pipeline_running:
    if st.session_state.pipeline_exit_code == 0:
        st.sidebar.success("✅ Pipeline finalizado correctamente.")
    else:
        st.sidebar.error(
            f"❌ El pipeline terminó con errores (código {st.session_state.pipeline_exit_code})."
        )

if st.session_state.pipeline_log:
    if st.sidebar.button("🗑️ Limpiar log", use_container_width=True):
        st.session_state.pipeline_log = ""
        st.session_state.pipeline_exit_code = None
        st.rerun()

st.sidebar.divider()

df_metricas = pd.read_sql("""
    SELECT ejecucion, condicion, modelo, auc_roc, gini, f1_score, accuracy, precision_m, recall, fecha_carga
    FROM metricas_modelo
    ORDER BY fecha_carga DESC
""", engine)

if df_metricas.empty:
    st.info("Todavía no hay métricas del modelo cargadas. Usa el botón **▶️ Ejecutar pipeline completo** en el panel lateral para generarlas.")
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
