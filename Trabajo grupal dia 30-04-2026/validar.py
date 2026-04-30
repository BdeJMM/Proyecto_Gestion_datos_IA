import logging
import pandas as pd

log = logging.getLogger(__name__)


def validar(df: pd.DataFrame) -> bool:
    """
    Validación post-limpieza del DataFrame.
    Recorre cada columna e informa sobre problemas encontrados.
    Retorna True si el DataFrame pasa todas las validaciones, False si hay errores.
    """
    log.info("── Validación iniciada ──")
    errores = []

    # ── 1. El DataFrame no está vacío ────────────────────────────────────────
    if df.empty:
        errores.append("El DataFrame está vacío")
        log.error("✗ El DataFrame está vacío")
    else:
        log.info(f"✓ DataFrame no vacío: {df.shape[0]} filas × {df.shape[1]} cols")

    # ── 2. Sin nulos ─────────────────────────────────────────────────────────
    nulos = df.isnull().sum()
    cols_con_nulos = nulos[nulos > 0]
    if not cols_con_nulos.empty:
        for col, n in cols_con_nulos.items():
            msg = f"Columna '{col}' tiene {n} nulos"
            errores.append(msg)
            log.error(f"✗ {msg}")
    else:
        log.info("✓ Sin nulos en ninguna columna")

    # ── 3. Sin duplicados ────────────────────────────────────────────────────
    duplicados = df.duplicated().sum()
    if duplicados > 0:
        msg = f"{duplicados} filas duplicadas encontradas"
        errores.append(msg)
        log.error(f"✗ {msg}")
    else:
        log.info("✓ Sin filas duplicadas")

    # ── 4. Validación por tipo de columna ────────────────────────────────────
    for col in df.columns:
        serie = df[col]

        if pd.api.types.is_numeric_dtype(serie):
            _validar_numerico(serie, col, errores)

        elif pd.api.types.is_datetime64_any_dtype(serie):
            _validar_fecha(serie, col, errores)

        else:
            _validar_texto(serie, col, errores)

    # ── Resultado final ──────────────────────────────────────────────────────
    if errores:
        log.error(f"── Validación FALLIDA — {len(errores)} problema(s) encontrado(s) ──")
        for e in errores:
            log.error(f"   • {e}")
        return False

    log.info("── Validación EXITOSA — todos los controles pasaron ──")
    return True


# ── Helpers ──────────────────────────────────────────────────────────────────

def _validar_numerico(serie: pd.Series, col: str, errores: list) -> None:
    """Verifica que no haya infinitos ni valores completamente constantes."""
    import numpy as np

    infinitos = serie.isin([float("inf"), float("-inf")]).sum()
    if infinitos > 0:
        msg = f"'{col}' tiene {infinitos} valores infinitos"
        errores.append(msg)
        log.error(f"✗ {msg}")
    else:
        log.info(f"✓ {col} [numérico] — sin infinitos | rango: [{serie.min():.2f}, {serie.max():.2f}]")

    if serie.nunique() == 1:
        log.warning(f"⚠ {col} [numérico] — columna constante (valor único: {serie.iloc[0]})")


def _validar_fecha(serie: pd.Series, col: str, errores: list) -> None:
    """Verifica que las fechas sean coherentes (no futuras si se esperan históricas)."""
    fecha_min = serie.min()
    fecha_max = serie.max()
    log.info(f"✓ {col} [fecha] — rango: [{fecha_min.date()}, {fecha_max.date()}]")

    fechas_futuras = (serie > pd.Timestamp.now()).sum()
    if fechas_futuras > 0:
        log.warning(f"⚠ {col} [fecha] — {fechas_futuras} fechas en el futuro")


def _validar_texto(serie: pd.Series, col: str, errores: list) -> None:
    """Verifica que no haya strings vacíos ni valores 'nan' como texto."""
    vacios = (serie.astype(str).str.strip() == "").sum()
    nan_texto = (serie.astype(str).str.lower() == "nan").sum()

    if vacios > 0:
        msg = f"'{col}' tiene {vacios} strings vacíos"
        errores.append(msg)
        log.error(f"✗ {msg}")

    if nan_texto > 0:
        msg = f"'{col}' tiene {nan_texto} valores 'nan' como texto"
        errores.append(msg)
        log.error(f"✗ {msg}")

    if vacios == 0 and nan_texto == 0:
        log.info(f"✓ {col} [texto] — {serie.nunique()} valores únicos, sin vacíos")
