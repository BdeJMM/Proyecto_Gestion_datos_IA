"""
pipeline.py
===========
Orquestador principal. Llama en orden a:

  1. Ingesta_EV2   → ingesta()   retorna df
  2. limpieza_EV2  → limpiar()   retorna df limpio
  3. validacion_EV2 → validar()  retorna bool
  4. train         → entrenar()  retorna TrainResult
  5. export_sql    → exportar()  retorna ExportResult

Si cualquier etapa falla el pipeline se detiene, registra el error
y retorna exit code 1 (GitHub Actions lo marca en rojo automáticamente).

Uso desde terminal o GitHub Actions:
  python src/pipeline.py
  python src/pipeline.py --fuente data/raw/bank.csv --db-url sqlite:///data/bank.db
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Carpetas mínimas — deben existir antes del FileHandler ───────────────────
for _d in ("reports", "data/raw", "data/processed", "data/backup", "models"):
    Path(_d).mkdir(parents=True, exist_ok=True)

# ── Logging estructurado ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("reports/pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("pipeline")


# ── Configuración de rutas por defecto ────────────────────────────────────────
FUENTE   = Path("data/raw/bank.csv")
DESTINO  = Path("data/processed/bank_work.csv")
RESPALDO = Path("data/backup/bank_backup.csv")
MODELO   = Path("models/model.pkl")


# ── Resultado global ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    ok: bool
    df_final: Optional[pd.DataFrame] = None
    metricas: dict = field(default_factory=dict)
    errores: list[str] = field(default_factory=list)
    etapa_fallida: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok

    def resumen(self) -> str:
        estado = "OK" if self.ok else f"FALLIDO en '{self.etapa_fallida}'"
        lineas = [
            "═" * 55,
            f"  Pipeline        : {estado}",
            f"  Errores         : {len(self.errores)}",
        ]
        if self.metricas:
            lineas += [
                f"  AUC-ROC (CV)    : {self.metricas.get('cv_auc_mean', 'n/a')}",
                f"  AUC-ROC (total) : {self.metricas.get('auc_roc', 'n/a')}",
                f"  F1-score        : {self.metricas.get('f1', 'n/a')}",
                f"  Precision       : {self.metricas.get('precision', 'n/a')}",
                f"  Recall          : {self.metricas.get('recall', 'n/a')}",
                f"  Filas           : {self.metricas.get('n_filas', 'n/a')}",
            ]
        if self.errores:
            lineas.append("  Errores detalle :")
            for e in self.errores:
                lineas.append(f"    • {e}")
        lineas.append("═" * 55)
        return "\n".join(lineas)


# ── Importación segura de módulos EV2 ────────────────────────────────────────
# Ingesta_EV2 tiene un typo (pahtlib) y variables sin definir.
# Lo parcheamos aquí sin tocar el archivo original.

def _importar_modulos():
    """
    Importa los módulos EV2 con manejo de sus errores conocidos.
    Retorna (ingesta_mod, limpieza_mod, validacion_mod) o lanza ImportError.
    """
    import types

    # ── Patch de Ingesta_EV2: corregir 'pahtlib' → 'pathlib' en runtime ──────
    src = Path(__file__).parent / "Ingesta_EV2.py"
    codigo = src.read_text(encoding="utf-8")
    codigo = codigo.replace("from pahtlib import Path", "from pathlib import Path")

    ingesta_mod = types.ModuleType("Ingesta_EV2")
    exec(compile(codigo, str(src), "exec"), ingesta_mod.__dict__)

    # limpieza y validacion se importan normalmente
    sys.path.insert(0, str(Path(__file__).parent))
    import limpieza_EV2 as limpieza_mod
    import validacion_EV2 as validacion_mod

    return ingesta_mod, limpieza_mod, validacion_mod


# ── Orquestador ───────────────────────────────────────────────────────────────

def ejecutar(
    fuente:   str | Path = FUENTE,
    destino:  str | Path = DESTINO,
    respaldo: str | Path = RESPALDO,
    ruta_modelo: str | Path = MODELO,
    db_url:   Optional[str] = None,
) -> PipelineResult:
    """
    Ejecuta el pipeline completo en 5 etapas.

    Parámetros
    ----------
    fuente      : CSV original (el 02_bank.csv que te dieron).
    destino     : copia de trabajo en data/processed/.
    respaldo    : copia de seguridad en data/backup/.
    ruta_modelo : dónde guardar el .pkl entrenado.
    db_url      : URL de SQLAlchemy. None → usa variable de entorno DB_URL
                  o SQLite local como fallback.
    """
    Path("reports").mkdir(parents=True, exist_ok=True)
    errores_globales: list[str] = []

    # ── Importar módulos EV2 ──────────────────────────────────────────────────
    try:
        ingesta_mod, limpieza_mod, validacion_mod = _importar_modulos()
        log.info("✓ Módulos EV2 cargados correctamente")
    except Exception as exc:
        msg = f"Error importando módulos EV2: {exc}"
        log.error(f"✗ {msg}")
        return PipelineResult(ok=False, errores=[msg], etapa_fallida="import")

    # ── Importar train y export_sql ───────────────────────────────────────────
    try:
        from entrenar_EV2 import entrenar
        from exportar_EV2 import exportar
    except ImportError as exc:
        msg = f"Error importando train/export_sql: {exc}"
        log.error(f"✗ {msg}")
        return PipelineResult(ok=False, errores=[msg], etapa_fallida="import")

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 1 — Ingesta
    # ════════════════════════════════════════════════════════════════════════
    log.info("▶ Etapa 1/5: Ingesta")
    df = None
    try:
        # Ingesta_EV2.ingesta() espera las variables en su scope global
        ingesta_mod.fuente   = str(fuente)
        ingesta_mod.destino  = str(destino)
        ingesta_mod.respaldo = str(respaldo)

        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        Path(respaldo).parent.mkdir(parents=True, exist_ok=True)

        df = ingesta_mod.ingesta()
        log.info(f"✓ Ingesta completa: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    except FileNotFoundError as exc:
        msg = f"Archivo no encontrado: {exc}"
        log.error(f"✗ {msg}")
        return PipelineResult(ok=False, errores=[msg], etapa_fallida="ingesta")
    except Exception as exc:
        msg = f"Error inesperado en ingesta: {type(exc).__name__}: {exc}"
        log.error(f"✗ {msg}")
        return PipelineResult(ok=False, errores=[msg], etapa_fallida="ingesta")

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 2 — Limpieza
    # ════════════════════════════════════════════════════════════════════════
    log.info("▶ Etapa 2/5: Limpieza")
    df_limpio = None
    try:
        df_limpio = limpieza_mod.limpiar(df)
        log.info(f"✓ Limpieza completa: {df_limpio.shape[0]:,} filas, {df_limpio.shape[1]} columnas")
    except Exception as exc:
        msg = f"Error inesperado en limpieza: {type(exc).__name__}: {exc}"
        log.error(f"✗ {msg}")
        return PipelineResult(ok=False, errores=[msg], etapa_fallida="limpieza")

    if df_limpio is None or df_limpio.empty:
        msg = "La limpieza devolvió un DataFrame vacío"
        log.error(f"✗ {msg}")
        return PipelineResult(ok=False, errores=[msg], etapa_fallida="limpieza")

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 3 — Validación
    # ════════════════════════════════════════════════════════════════════════
    log.info("▶ Etapa 3/5: Validación")
    try:
        valido = validacion_mod.validar(df_limpio)
    except Exception as exc:
        msg = f"Error inesperado en validación: {type(exc).__name__}: {exc}"
        log.error(f"✗ {msg}")
        return PipelineResult(ok=False, errores=[msg], etapa_fallida="validacion")

    if not valido:
        msg = "El DataFrame no pasó la validación — revisa el log para ver qué columnas fallaron"
        log.error(f"✗ {msg}")
        return PipelineResult(ok=False, errores=[msg], etapa_fallida="validacion")

    log.info("✓ Validación aprobada")

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 4 — Entrenamiento
    # ════════════════════════════════════════════════════════════════════════
    log.info("▶ Etapa 4/5: Entrenamiento XGBoost")
    try:
        resultado_train = entrenar(df_limpio, ruta_modelo=ruta_modelo)
    except Exception as exc:
        msg = f"Error inesperado en entrenamiento: {type(exc).__name__}: {exc}"
        log.error(f"✗ {msg}")
        return PipelineResult(ok=False, errores=[msg], etapa_fallida="entrenamiento")

    if not resultado_train.ok:
        return PipelineResult(
            ok=False,
            errores=resultado_train.errores,
            etapa_fallida="entrenamiento",
        )

    log.info(f"✓ Modelo listo — AUC-ROC CV: {resultado_train.metricas.get('cv_auc_mean')}")

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 5 — Exportación SQL
    # ════════════════════════════════════════════════════════════════════════
    log.info("▶ Etapa 5/5: Exportación SQL")
    try:
        resultado_export = exportar(
            df_clientes=df_limpio,
            df_predicciones=resultado_train.df_predicciones,
            metricas=resultado_train.metricas,
            db_url=db_url,
        )
    except Exception as exc:
        msg = f"Error inesperado en exportación SQL: {type(exc).__name__}: {exc}"
        log.error(f"✗ {msg}")
        # La exportación falla pero el modelo ya fue entrenado → advertencia, no fallo total
        errores_globales.append(msg)
        log.warning("⚠ Pipeline completo pero exportación SQL falló")
        return PipelineResult(
            ok=True,          # el modelo es válido aunque no se exportó
            df_final=df_limpio,
            metricas=resultado_train.metricas,
            errores=errores_globales,
        )

    if not resultado_export.ok:
        errores_globales.extend(resultado_export.errores)
        log.warning(f"⚠ Exportación SQL tuvo errores: {resultado_export.errores}")

    log.info(f"✓ Tablas exportadas: {resultado_export.tablas_creadas}")

    return PipelineResult(
        ok=True,
        df_final=df_limpio,
        metricas=resultado_train.metricas,
        errores=errores_globales,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline Bank Marketing ML")
    p.add_argument("--fuente",    default=str(FUENTE),   help="Ruta al CSV original")
    p.add_argument("--destino",   default=str(DESTINO),  help="Ruta copia de trabajo")
    p.add_argument("--respaldo",  default=str(RESPALDO), help="Ruta del respaldo")
    p.add_argument("--modelo",    default=str(MODELO),   help="Ruta para guardar el .pkl")
    p.add_argument("--db-url",    default=None,
                   help="URL SQLAlchemy. Ej: sqlite:///data/bank.db | "
                        "postgresql+psycopg2://user:pass@host/db | "
                        "oracle+cx_oracle://user:pass@host:1521/ORCL")
    return p.parse_args()


if __name__ == "__main__":
    args = _args()

    resultado = ejecutar(
        fuente=args.fuente,
        destino=args.destino,
        respaldo=args.respaldo,
        ruta_modelo=args.modelo,
        db_url=args.db_url,
    )

    print(resultado.resumen())
    sys.exit(0 if resultado.ok else 1)
