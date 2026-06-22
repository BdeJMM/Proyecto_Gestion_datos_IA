import logging
import pandas as pd
from pathlib import Path

from ingesta    import ingestar
from limpiar   import limpiar
from validar import validar
from guardado import guardar

# ── Logging ──────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Configuración ─────────────────────────────────────────────────────────────
FUENTE_PATH    = "data.csv"          # ← CSV de origen (ajusta aquí)
RAW_PATH       = "data/raw/data.csv"
BACKUP_PATH    = "data/backup/data.csv"
PROCESSED_PATH = "data/processed/data_clean.csv"
ID_COL         = "UserID"


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMACIÓN (columnas derivadas + estandarización)
# ══════════════════════════════════════════════════════════════════════════════

def transformar(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza formatos y crea columnas derivadas."""
    log.info("── Transformación iniciada ──")
    df = df.copy()

    # Redondear floats a 2 decimales
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].round(2)
        log.info(f"  {col} [float] — redondeado a 2 decimales")

    # Columnas derivadas
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 29, 44, 100],
        labels=["Young (18-29)", "Middle (30-44)", "Senior (45+)"]
    )
    log.info("  Nueva columna: AgeGroup")

    df["DurationCategory"] = pd.cut(
        df["Duration"],
        bins=[0, 20, 40, float("inf")],
        labels=["Short (<20 min)", "Medium (20-40 min)", "Long (>40 min)"]
    )
    log.info("  Nueva columna: DurationCategory")

    immersion_norm = (df["ImmersionLevel"] - 1) / 4
    sickness_norm  = (df["MotionSickness"] - 1) / 9
    df["ComfortScore"] = ((immersion_norm - sickness_norm + 1) / 2 * 10).round(2)
    log.info("  Nueva columna: ComfortScore")

    log.info("── Transformación finalizada ──")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 55)
    log.info("INICIO DEL PIPELINE")
    log.info("=" * 55)

    # 1. Ingesta
    log.info("\n[1/4] INGESTA")
    df = ingestar(FUENTE_PATH, RAW_PATH, BACKUP_PATH)

    # 2. Limpieza
    log.info("\n[2/4] LIMPIEZA")
    df = limpiar(df, id_col=ID_COL)

    # 3. Transformación
    log.info("\n[3/4] TRANSFORMACIÓN")
    df = transformar(df)

    # 4. Validación
    log.info("\n[4/4] VALIDACIÓN")
    ok = validar(df)

    if not ok:
        log.error("Pipeline detenido — el DataFrame no pasó la validación.")
        log.error("Revisar logs/pipeline.log para más detalles.")
        raise RuntimeError("Validación fallida.")

    # Guardar solo si la validación fue exitosa
    guardar(df, PROCESSED_PATH)

    log.info("\n" + "=" * 55)
    log.info("PIPELINE COMPLETADO EXITOSAMENTE")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
