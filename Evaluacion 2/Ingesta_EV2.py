import pandas as pd
import shutil 
import logging

from pahtlib import Path

log = logging.getLogger(__name__)

def ingesta():

    log.info("── Ingesta iniciada ──")
    
    
    if not Path(fuente).exists():
        raise FileNotFoundError(f"No se encontró el archivo de origen: {fuente}")
    log.info(f"✓ Archivo de origen encontrado: {fuente}")
    
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fuente, destino)
    log.info(f"CSV copiado en: {destino}")


    Path(respaldo).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fuente, respaldo)
    log.info(f"✓ Respaldo guardado en: {respaldo}")

    df = pd.read_csv(destino, encoding="utf-8-sig")

    if df.empty:
        raise ValueError("El archivo CSV no contiene datos")

    log.info(f"✓ Filas cargadas: {len(df)} | Columnas: {list(df.columns)}")
    log.info("── Ingesta finalizada ──")
    return df

