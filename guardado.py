import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

DIR_VALIDOS = Path("data/processed")
DIR_ERRORES = Path("data/errores")

# Mapeo de columnas del DataFrame → nombres en Oracle
# (evita palabras reservadas: DEFAULT → def_credit, DAY → day_contact)
_RENAME_ORACLE = {
    "default": "def_credit",
    "day":     "day_contact",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _calcular_completitud(filas_entrada, filas_salida):
    if filas_entrada and filas_entrada > 0 and filas_salida is not None:
        return round(100 * filas_salida / filas_entrada, 2)
    return None


def _preparar_df(df: pd.DataFrame, ejecucion: str, condicion: str) -> pd.DataFrame:
    """Añade columnas de control y renombra las reservadas para Oracle."""
    d = df.copy().rename(columns=_RENAME_ORACLE)
    d.insert(0, "condicion", condicion)
    d.insert(0, "ejecucion", ejecucion)
    return d


# ── Guardado en disco ─────────────────────────────────────────────────────────

def guardar_csv(df_validos: pd.DataFrame, df_errores: pd.DataFrame) -> dict:
    """Persiste los DataFrames en disco como CSV con timestamp."""
    DIR_VALIDOS.mkdir(parents=True, exist_ok=True)
    DIR_ERRORES.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_validos = DIR_VALIDOS / f"bank_validos_{ts}.csv"
    ruta_errores = DIR_ERRORES / f"bank_errores_{ts}.csv"

    df_validos.to_csv(ruta_validos, index=False, encoding="utf-8-sig")
    log.info(f"CSV válidos  → {ruta_validos}  ({len(df_validos)} filas)")

    ruta_err_str = None
    if len(df_errores) > 0:
        df_errores.to_csv(ruta_errores, index=False, encoding="utf-8-sig")
        log.info(f"CSV erróneos → {ruta_errores}  ({len(df_errores)} filas)")
        ruta_err_str = str(ruta_errores)
    else:
        log.info("Sin registros erróneos que guardar en disco.")

    return {
        "ruta_validos": str(ruta_validos),
        "ruta_errores": ruta_err_str,
        "n_validos":    len(df_validos),
        "n_errores":    len(df_errores),
    }


# ── Guardado en Oracle ────────────────────────────────────────────────────────

def guardar_en_oracle(
    engine:           Engine,
    df_validos:       pd.DataFrame,
    df_errores:       pd.DataFrame,
    resultado_modelo: dict,
    lista_metricas:   list,
    condicion:        str,
    ejecucion:        str,
) -> None:
    """
    Inserta todo en Oracle en una sola transacción:
      • clientes_validos
      • registros_erroneos  (solo si hay errores)
      • metricas_modelo     (baseline + xgboost)
      • rendimiento_pipeline
    """
    log.info(f"Oracle | iniciando guardado — ejecución={ejecucion}  condición={condicion}")

    with engine.begin() as conn:  # begin() hace commit automático al salir sin error

        # ── 1. Clientes válidos ───────────────────────────────────────────────
        df_v = _preparar_df(df_validos, ejecucion, condicion)
        df_v.to_sql("clientes_validos", conn, if_exists="append", index=False)
        log.info(f"Oracle | clientes_validos: {len(df_v)} filas")

        # ── 2. Registros erróneos ─────────────────────────────────────────────
        if len(df_errores) > 0:
            df_e = _preparar_df(df_errores, ejecucion, condicion)
            # fila_original proviene del índice si existe la columna motivo_error
            if "motivo_error" in df_e.columns and df_e.index.name != "fila_original":
                df_e.insert(2, "fila_original", df_e.index)
            df_e.to_sql("registros_erroneos", conn, if_exists="append", index=False)
            log.info(f"Oracle | registros_erroneos: {len(df_e)} filas")
        else:
            log.info("Oracle | sin registros erróneos que insertar")

        # ── 3. Métricas del modelo ────────────────────────────────────────────
        filas_modelo = [
            {
                "ejecucion":   ejecucion,
                "condicion":   condicion,
                "modelo":      nombre,
                "accuracy":    round(resultado_modelo[clave]["accuracy"],  4),
                "precision_m": round(resultado_modelo[clave]["precision"], 4),
                "recall":      round(resultado_modelo[clave]["recall"],    4),
                "f1_score":    round(resultado_modelo[clave]["f1"],        4),
                "auc_roc":     round(resultado_modelo[clave]["auc_roc"],   4),
            }
            for clave, nombre in [
                ("baseline", "Regresión Logística"),
                ("xgboost",  "XGBoost"),
            ]
        ]
        pd.DataFrame(filas_modelo).to_sql(
            "metricas_modelo", conn, if_exists="append", index=False
        )
        log.info("Oracle | metricas_modelo: 2 filas (baseline + xgboost)")

        # ── 4. Rendimiento del pipeline ───────────────────────────────────────
        filas_rend = [
            {
                "ejecucion":     ejecucion,
                "condicion":     condicion,
                "etapa":         m.etapa,
                "latencia_seg":  m.latencia_seg,
                "filas_entrada": m.filas_entrada,
                "filas_salida":  m.filas_salida,
                "completitud":   _calcular_completitud(m.filas_entrada, m.filas_salida),
                "exito":         "Si" if m.exito else "No",
                "error":         m.error or "",
            }
            for m in lista_metricas if m is not None
        ]
        if filas_rend:
            pd.DataFrame(filas_rend).to_sql(
                "rendimiento_pipeline", conn, if_exists="append", index=False
            )
            log.info(f"Oracle | rendimiento_pipeline: {len(filas_rend)} etapas")
        else:
            log.warning("Oracle | sin métricas de rendimiento para guardar")

    log.info(f"Oracle | guardado completo — ejecución={ejecucion}")


# ── Función de compatibilidad para pipeline.py (solo CSV) ────────────────────

def guardar(df_validos: pd.DataFrame, df_errores: pd.DataFrame) -> dict:
    """Interfaz original — guarda solo en CSV."""
    log.info("-- Guardado CSV iniciado --")
    resultado = guardar_csv(df_validos, df_errores)
    log.info("-- Guardado CSV finalizado --")
    return resultado
