import logging
import pandas as pd

log = logging.getLogger(__name__)


def limpiar(df: pd.DataFrame, id_col: str = None) -> pd.DataFrame:
    """
    Limpieza genérica que detecta automáticamente el tipo de cada columna
    y aplica la limpieza correspondiente.

    - Columnas numéricas : convierte a numérico, imputa nulos con la mediana.
    - Columnas de texto  : strip + title case, elimina filas con valor vacío.
    - Columnas de fecha  : convierte a datetime, elimina filas con fecha inválida.
    - Duplicados         : elimina por id_col si se indica, si no por fila completa.
    """
    log.info("── Limpieza iniciada ──")
    df = df.copy()

    log.info(f"Shape inicial: {df.shape}")
    log.info(f"Nulos por columna:\n{df.isnull().sum().to_string()}")
    log.info(f"Filas duplicadas: {df.duplicated().sum()}")

    for col in df.columns:
        serie = df[col]

        # ── Detección de tipo ────────────────────────────────────────────────
        es_fecha    = _detectar_fecha(serie)
        es_numerico = pd.api.types.is_numeric_dtype(serie) or _intentar_numerico(serie)

        if es_fecha:
            _limpiar_fecha(df, col)
        elif es_numerico:
            _limpiar_numerico(df, col)
        else:
            _limpiar_texto(df, col)

    # ── Duplicados ───────────────────────────────────────────────────────────
    antes = len(df)
    subset = [id_col] if id_col and id_col in df.columns else None
    df.drop_duplicates(subset=subset, inplace=True)
    log.info(f"Duplicados — eliminadas {antes - len(df)} filas "
             f"({'por ' + id_col if subset else 'por fila completa'})")

    df.reset_index(drop=True, inplace=True)
    log.info(f"Shape final: {df.shape}")
    log.info("── Limpieza finalizada ──")

    return df


# ── Helpers de detección ─────────────────────────────────────────────────────

def _detectar_fecha(serie: pd.Series) -> bool:
    """Detecta si una columna parece contener fechas."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        return True
    if pd.api.types.is_numeric_dtype(serie):
        return False
    muestra = serie.dropna().astype(str).head(20)
    convertidas = pd.to_datetime(muestra, errors="coerce", format="mixed").notna().sum()
    return convertidas / max(len(muestra), 1) >= 0.8


def _intentar_numerico(serie: pd.Series) -> bool:
    """Detecta si una columna de texto puede convertirse a numérico."""
    muestra = serie.dropna().astype(str).str.strip().head(20)
    convertidas = pd.to_numeric(muestra, errors="coerce").notna().sum()
    return convertidas / max(len(muestra), 1) >= 0.8


# ── Helpers de limpieza ──────────────────────────────────────────────────────

def _limpiar_numerico(df: pd.DataFrame, col: str) -> None:
    """Convierte a numérico e imputa nulos con la mediana."""
    df[col] = pd.to_numeric(df[col], errors="coerce")
    nulos = df[col].isnull().sum()
    if nulos > 0:
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)
        log.info(f"{col} [numérico] — imputadas {nulos} filas con mediana ({mediana:.2f})")
    else:
        log.info(f"{col} [numérico] — sin nulos, convertido correctamente")


def _limpiar_texto(df: pd.DataFrame, col: str) -> None:
    """Strip + title case; elimina filas con valor vacío o nulo."""
    df[col] = df[col].astype(str).str.strip().replace("", pd.NA)
    df[col] = df[col].where(df[col].str.lower() != "nan", pd.NA)
    df[col] = df[col].str.title()
    antes = len(df)
    df.dropna(subset=[col], inplace=True)
    log.info(f"{col} [texto] — eliminadas {antes - len(df)} filas vacías/nulas")


def _limpiar_fecha(df: pd.DataFrame, col: str) -> None:
    """Convierte a datetime y elimina filas con fecha inválida."""
    df[col] = pd.to_datetime(df[col], errors="coerce")
    antes = len(df)
    df.dropna(subset=[col], inplace=True)
    log.info(f"{col} [fecha] — eliminadas {antes - len(df)} filas con fecha inválida")
