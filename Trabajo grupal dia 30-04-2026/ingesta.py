import shutil
import logging
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)


def ingestar(fuente: str, destino: str, respaldo: str) -> pd.DataFrame:
    """
    Copia el CSV desde `fuente` hacia data/raw/ y hace un respaldo.
    Solo carga el archivo tal como está — la limpieza la hace limpieza.py.
    """
    log.info("── Ingesta iniciada ──")

    # ── Validar que la fuente exista ─────────────────────────────────────────
    if not Path(fuente).exists():
        raise FileNotFoundError(f"No se encontró el archivo de origen: {fuente}")
    log.info(f"✓ Archivo de origen encontrado: {fuente}")

    # ── Copiar a data/raw/ ───────────────────────────────────────────────────
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fuente, destino)
    log.info(f"✓ CSV copiado a: {destino}")

    # ── Respaldo ─────────────────────────────────────────────────────────────
    Path(respaldo).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fuente, respaldo)
    log.info(f"✓ Respaldo guardado en: {respaldo}")

    # ── Carga ────────────────────────────────────────────────────────────────
    df = pd.read_csv(destino, encoding="utf-8-sig")

    if df.empty:
        raise ValueError("El archivo CSV no contiene datos")

    log.info(f"✓ Filas cargadas: {len(df)} | Columnas: {list(df.columns)}")
    log.info("── Ingesta finalizada ──")
    return df
