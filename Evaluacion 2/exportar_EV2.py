"""
export_sql.py
=============
Exporta los datos limpios y las predicciones del modelo a SQL.
Soporta SQLite (local), Oracle Cloud y Amazon RDS sin cambiar el código —
solo cambia la variable de entorno DB_URL.

Llamado por pipeline.py después de entrenar.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text, Engine
from sqlalchemy.exc import SQLAlchemyError

log = logging.getLogger(__name__)


# ── Resultado tipado ──────────────────────────────────────────────────────────

@dataclass
class ExportResult:
    ok: bool
    tablas_creadas: list[str] = field(default_factory=list)
    filas_exportadas: dict[str, int] = field(default_factory=dict)
    errores: list[str] = field(default_factory=list)
    db_url_tipo: str = ""   # "sqlite" | "oracle" | "postgresql" | "mysql"

    def __bool__(self) -> bool:
        return self.ok


# ── Engine — una sola función, todos los destinos ────────────────────────────

def _get_engine(db_url: Optional[str] = None) -> Engine:
    """
    Construye el engine SQLAlchemy según DB_URL.

    Prioridad:
      1. Argumento db_url (tests o llamada directa)
      2. Variable de entorno DB_URL
      3. Fallback: SQLite local en data/bank.db

    Ejemplos de DB_URL:
      SQLite local   → sqlite:///data/bank.db
      Oracle Cloud   → oracle+cx_oracle://user:pass@host:1521/?service_name=ORCL
      Amazon RDS PG  → postgresql+psycopg2://user:pass@host:5432/bank
      Amazon RDS MySQL → mysql+pymysql://user:pass@host:3306/bank
    """
    url = db_url or os.getenv("DB_URL", "sqlite:///data/bank.db")
    engine = create_engine(url, pool_pre_ping=True, echo=False)
    log.info(f"✓ Engine creado: {url.split('://')[0]}")   # no loguea credenciales
    return engine


# ── Función principal ─────────────────────────────────────────────────────────

def exportar(
    df_clientes: pd.DataFrame,
    df_predicciones: pd.DataFrame,
    metricas: dict,
    *,
    db_url: Optional[str] = None,
) -> ExportResult:
    """
    Escribe tres tablas en la base de datos:

    bank_clientes      → datos limpios originales (replace en cada run)
    bank_predicciones  → prob_yes, prediccion, real, correcto, run_ts
    bank_runs          → historial de métricas por run (append, nunca se borra)

    Parámetros
    ----------
    df_clientes    : DataFrame limpio (salida de limpiar).
    df_predicciones: DataFrame de predicciones (salida de entrenar).
    metricas       : dict con auc_roc, f1, precision, recall, etc.
    db_url         : URL de conexión. Si es None, usa la variable de entorno DB_URL.
    """
    log.info("── Exportación SQL iniciada ──")

    # ── Guardias de entrada ───────────────────────────────────────────────────
    errores: list[str] = []

    if not isinstance(df_clientes, pd.DataFrame) or df_clientes.empty:
        msg = "df_clientes no es un DataFrame válido o está vacío"
        log.error(f"✗ {msg}")
        return ExportResult(ok=False, errores=[msg])

    if not isinstance(df_predicciones, pd.DataFrame) or df_predicciones.empty:
        msg = "df_predicciones no es un DataFrame válido o está vacío"
        log.error(f"✗ {msg}")
        return ExportResult(ok=False, errores=[msg])

    # ── Conexión ──────────────────────────────────────────────────────────────
    try:
        engine = _get_engine(db_url)
        db_tipo = engine.dialect.name   # sqlite | oracle | postgresql | mysql
    except Exception as exc:
        msg = f"No se pudo crear el engine de base de datos: {exc}"
        log.error(f"✗ {msg}")
        return ExportResult(ok=False, errores=[msg])

    tablas_creadas: list[str] = []
    filas_exportadas: dict[str, int] = {}

    # ── Tabla 1: clientes limpios ─────────────────────────────────────────────
    try:
        with engine.begin() as conn:
            df_clientes.to_sql(
                "bank_clientes", conn,
                if_exists="replace",   # sobreescribe en cada campaña nueva
                index=False,
                chunksize=1000,        # evita timeouts en conexiones cloud
            )
        tablas_creadas.append("bank_clientes")
        filas_exportadas["bank_clientes"] = len(df_clientes)
        log.info(f"✓ bank_clientes — {len(df_clientes):,} filas")
    except SQLAlchemyError as exc:
        msg = f"Error exportando bank_clientes: {exc}"
        log.error(f"✗ {msg}")
        errores.append(msg)

    # ── Tabla 2: predicciones ─────────────────────────────────────────────────
    try:
        with engine.begin() as conn:
            df_predicciones.to_sql(
                "bank_predicciones", conn,
                if_exists="replace",
                index=False,
                chunksize=1000,
            )
        tablas_creadas.append("bank_predicciones")
        filas_exportadas["bank_predicciones"] = len(df_predicciones)
        log.info(f"✓ bank_predicciones — {len(df_predicciones):,} filas")
    except SQLAlchemyError as exc:
        msg = f"Error exportando bank_predicciones: {exc}"
        log.error(f"✗ {msg}")
        errores.append(msg)

    # ── Tabla 3: historial de runs (append, nunca replace) ────────────────────
    try:
        df_run = pd.DataFrame([{
            "run_ts":      pd.Timestamp.now(),
            "modelo":      "xgboost",
            "n_filas":     metricas.get("n_filas"),
            "n_features":  metricas.get("n_features"),
            "auc_roc":     metricas.get("auc_roc"),
            "cv_auc_mean": metricas.get("cv_auc_mean"),
            "cv_auc_std":  metricas.get("cv_auc_std"),
            "precision":   metricas.get("precision"),
            "recall":      metricas.get("recall"),
            "f1":          metricas.get("f1"),
        }])
        with engine.begin() as conn:
            df_run.to_sql(
                "bank_runs", conn,
                if_exists="append",    # historial acumulativo
                index=False,
            )
        tablas_creadas.append("bank_runs")
        filas_exportadas["bank_runs"] = 1
        log.info(f"✓ bank_runs — run registrado (auc={metricas.get('auc_roc')})")
    except SQLAlchemyError as exc:
        msg = f"Error exportando bank_runs: {exc}"
        log.error(f"✗ {msg}")
        errores.append(msg)

    # ── Verificación: leer conteos de vuelta ──────────────────────────────────
    try:
        with engine.connect() as conn:
            for tabla in tablas_creadas:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {tabla}")).scalar()
                log.info(f"✓ Verificación {tabla}: {count:,} filas en BD")
    except Exception as exc:
        log.warning(f"⚠ No se pudo verificar conteos: {exc}")

    ok = len(errores) == 0
    estado = "finalizada sin errores" if ok else f"finalizada con {len(errores)} error(es)"
    log.info(f"── Exportación SQL {estado} ──")

    return ExportResult(
        ok=ok,
        tablas_creadas=tablas_creadas,
        filas_exportadas=filas_exportadas,
        errores=errores,
        db_url_tipo=db_tipo,
    )
