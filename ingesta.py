import shutil 
import logging
import pandas as pd 

from pathlib import Path

#Columnas esperadas
c_e=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','deposit']

log=logging.getLogger(__name__)

def ingestar(origen: str, destino: str):
    log.info(f"Ingesta iniciada — origen: {origen}")

    if not Path(origen).exists():
        log.error(f"Archivo no encontrado: {origen}")
        raise FileNotFoundError(f"No se encontró el archivo: {origen}")
    log.info("Validación 1: archivo existente")

    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(origen, destino)
    log.info(f"Respaldo guardado en: {destino}")

    df = pd.read_csv(origen, encoding="utf-8-sig", sep=",")
    log.info(f"Filas cargadas: {len(df)} | Columnas: {list(df.columns)}")

    faltantes = [c for c in c_e if c not in df.columns]
    if faltantes:
        log.error(f"Columnas faltantes: {faltantes}")
        raise ValueError(f"El archivo no tiene las columnas esperadas: {faltantes}")
    log.info("Validación 2: todas las columnas esperadas presentes")

    if df.empty:
        log.error("El archivo está vacío")
        raise ValueError("El archivo CSV no contiene datos")
    log.info("Validación 3: el DataFrame no está vacío")

    log.info(f"Nulos por columna:\n{df.isnull().sum().to_string()}")
    log.info(f"Filas duplicadas: {df.duplicated().sum()}")

    return df