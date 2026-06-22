import logging
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)


def guardar(df: pd.DataFrame, ruta: str) -> None:
    """Guarda el DataFrame procesado en data/processed/."""
    log.info("── Guardado iniciado ──")

    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8-sig")

    log.info(f"✓ Archivo guardado en: {ruta}")
    log.info(f"  Shape: {df.shape[0]} filas × {df.shape[1]} cols")
    log.info(f"  Columnas: {list(df.columns)}")
    log.info("── Guardado finalizado ──")
