import logging
import pandas as pd 
from ingesta import c_e

log = logging.getLogger(__name__)

def limpiar(df:pd.DataFrame,c_e) -> pd.DataFrame:
    log.info("-- INICIO DE LIMPIEZA --")
    df=df.copy()

    log.info(f"Shape inicial: {df.shape}")
    log.info(f"Nulos por columna:\n{df.isnull().sum().to_string()}")
    log.info(f"Filas duplicadas: {df.duplicated().sum()}")

    #Dicionario de tipo de columna
    c_t = {
    # Numéricas
    'age':      {'tipo': 'numeric', 'accion': 'median', 'depup': False}, 'balance':  {'tipo': 'numeric', 'accion': 'median', 'depup': False},
    'day': {'tipo': 'numeric', 'accion': 'median', 'depup': False},'duration': {'tipo': 'numeric', 'accion': 'median', 'depup': False},
    'campaign': {'tipo': 'numeric', 'accion': 'median', 'depup': False},'pdays': {'tipo': 'numeric', 'accion': 'median', 
    'depup': False}, 'previous': {'tipo': 'numeric', 'accion': 'median', 'depup': False},
    # Categóricas
    'job':      {'tipo': 'str', 'accion': 'drop', 'depup': False},'marital':  {'tipo': 'str', 'accion': 'mode', 'depup': False},
    'education':{'tipo': 'str', 'accion': 'drop', 'depup': False},'default':  {'tipo': 'str', 'accion': 'mode', 'depup': False},
    'housing':  {'tipo': 'str', 'accion': 'mode', 'depup': False},'loan':     {'tipo': 'str', 'accion': 'mode', 'depup': False},
    'contact':  {'tipo': 'str', 'accion': 'mode', 'depup': False},'month':    {'tipo': 'str', 'accion': 'mode', 'depup': False},
    'poutcome': {'tipo': 'str', 'accion': 'mode', 'depup': False},'deposit':  {'tipo': 'str', 'accion': 'mode', 'depup': False},
}
    #Columnas a limpiar
    c_l = {
        col: cfg #cfg es config
        for col, cfg in c_t.items()
        if col in c_e
    }

    for col, cfg in c_l.items():
       
        if col not in df.columns:
            log.warning(f"{col} — esperada pero ausente en el DataFrame, se omite")
            continue

        tipo   = cfg['tipo']
        accion = cfg['accion']

        if tipo == 'str':
            df[col] = df[col].astype(str).str.strip().replace('', pd.NA)

        elif tipo == 'numeric':
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip().replace('', pd.NA),
                errors='coerce'
            )

        elif tipo == 'datetime':
            df[col] = pd.to_datetime(df[col], errors='coerce')

        nulos = df[col].isnull().sum()
        
        if accion == 'drop':
            antes = len(df)
            df.dropna(subset=[col], inplace=True)
            log.info(f"{col} ({tipo}) — eliminadas {antes - len(df)} filas nulas/inválidas")

        elif accion == 'median':
            df[col] = df[col].fillna(df[col].median())
            log.info(f"{col} ({tipo}) — imputadas {nulos} filas con la mediana")

        elif accion == 'mode':
            df[col] = df[col].fillna(df[col].mode().iloc[0])
            log.info(f"{col} ({tipo}) — imputadas {nulos} filas con la moda")

        elif accion == 'keep':
            log.info(f"{col} ({tipo}) — {nulos} nulos conservados (accion=keep)")

    claves_depup = [
        c for c, cfg in c_l.items()
        if cfg.get('depup') and c in df.columns
    ]
    if claves_depup:
        antes = len(df)
        df.drop_duplicates(subset=claves_depup, inplace=True)
        log.info(f"duplicados — eliminadas {antes - len(df)} filas por clave(s) {claves_depup}")

    df.reset_index(drop=True, inplace=True)
    log.info(f"Shape final: {df.shape}")
    log.info("-- Limpieza finalizada --")

    return df