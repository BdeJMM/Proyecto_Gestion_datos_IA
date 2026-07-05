import logging
import pandas as pd
 
log = logging.getLogger(__name__)
 
# Valores categoricos y binarios

v_p = {
    # Variables categóricas
    "job":       {"admin.", "blue-collar", "entrepreneur", "housemaid",
                  "management", "retired", "self-employed", "services",
                  "student", "technician", "unemployed", "unknown"},
    "month":     {"jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec"},
    "education": {"primary", "secondary", "tertiary", "unknown"},
    "poutcome":  {"success", "failure", "other", "unknown"},
    "contact":   {"cellular", "telephone", "unknown"},
    "marital":   {"married", "single", "divorced"},
    # Variables binarias (sí/no)
    "default":   {"yes", "no"},
    "housing":   {"yes", "no"},
    "loan":      {"yes", "no"},
    "deposit":   {"yes", "no"}
}
 
# Rangos numericos

rang = {
    # Datos demográficos
    "age":      (18,   95,   "edad del cliente en años"),
    # Perfil financiero
    "balance":  (None, None, "saldo promedio en la cuenta"),
    # Historial de contacto — campaña actual
    "duration": (0,    None, "duración de llamada en segundos (≥ 0)"),
    "day":      (1,    31,   "día del mes del último contacto"),
    "campaign": (1,    None,   "nº de contactos en la campaña actual"),
    # Historial de contacto — campañas anteriores
    "pdays":    (-1,   None, "días desde último contacto previo (-1 = sin contacto)"),
    "previous": (0,    None, "nº de contactos antes de la campaña actual"),
}
 
 
def validar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
 
    log.info("-- Validación iniciada --")
 
    df = df.copy()
    df["_errores"] = ""

    # ── Validación categórica ─────────────────────────────────────────────────
    for i, (col, valores) in enumerate(v_p.items(), start=1):
        if col not in df.columns:
            log.warning(f"Semántica str {i:02} — columna '{col}' ausente, se omite")
            continue
        mascara = ~df[col].str.lower().str.strip().isin(valores)
        df.loc[mascara, "_errores"] += f"{col} con valor no reconocido; "
        log.info(f"Semántica str {i:02} — {col} no reconocido: {mascara.sum()} registros")
 
    # ── Validación numérica ───────────────────────────────────────────────────
    for i, (col, (minv, maxv, desc)) in enumerate(rang.items(), start=1):
        if col not in df.columns:
            log.warning(f"Semántica num {i:02} — columna '{col}' ausente, se omite")
            continue
        if minv is not None and maxv is not None:
            mascara = ~df[col].between(minv, maxv)
        elif minv is not None:
            mascara = df[col] < minv
        else:
            mascara = df[col] > maxv
        df.loc[mascara, "_errores"] += f"{col} fuera de rango ({desc}); "
        log.info(f"Semántica num {i:02} — {col} fuera de rango: {mascara.sum()} registros")
 
    # ── Validación de coherencia de negocio ───────────────────────────────────
    # R1: Si pdays == -1 (sin contacto previo), previous debe ser 0
    mascara = (df["pdays"] == -1) & (df["previous"] > 0)
    df.loc[mascara, "_errores"] += "incoherencia: pdays=-1 pero previous>0; "
    log.info(f"Coherencia R1 — pdays=-1 con previous>0: {mascara.sum()} registros")
 
    # R2: Si previous == 0, poutcome debería ser 'unknown'
    mascara = (df["previous"] == 0) & (df["poutcome"].str.lower().str.strip() != "unknown")
    df.loc[mascara, "_errores"] += "incoherencia: previous=0 pero poutcome no es unknown; "
    log.info(f"Coherencia R2 — previous=0 con poutcome distinto de unknown: {mascara.sum()} registros")
 
    # R3: duration=0 implica que no hubo contacto real — marcar como sospechoso
    mascara = df["duration"] == 0
    df.loc[mascara, "_errores"] += "duration=0 (sin contacto real registrado); "
    log.info(f"Coherencia R3 — duration=0: {mascara.sum()} registros")
 
    # ── Separación válidos / erróneos ─────────────────────────────────────────
    df_errores = df[df["_errores"] != ""].copy()
    df_validos  = df[df["_errores"] == ""].drop(columns=["_errores"])
    df_errores  = df_errores.rename(columns={"_errores": "motivo_error"})
 
    log.info(f"Registros válidos:  {len(df_validos)}")
    log.info(f"Registros erróneos: {len(df_errores)}")
    log.info("-- Validación finalizada --")
 
    return df_validos, df_errores