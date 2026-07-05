import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

from ingesta import ingestar, c_e
from limpieza import limpiar
from validacion import validar
from guardado import guardar_csv, guardar_en_oracle
from modelo import modelar
from rendimiento import medir_etapa, resumen_metricas
from conexion import get_engine
from tablas import crear_tablas

# ── Logging ───────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(
            f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

ORIGEN  = "02_bank.csv"
DESTINO = "data/raw/02_bank_backup.csv"


# ── Pipeline ──────────────────────────────────────────────────────────────────

def ejecutar_pipeline(
    sufijo:     str             = "",
    condicion:  str             = "",
    ejecucion:  str             = "",
    df_externo: pd.DataFrame    = None,
    engine                      = None,
    guardar_oracle: bool        = True,
) -> tuple[list, dict | None]:
    """
    Ejecuta el pipeline completo midiendo cada etapa.

    Parámetros
    ----------
    sufijo        : etiqueta legible para logs ("Dataset completo", "50%", …)
    condicion     : etiqueta corta para Oracle ("completo", "50pct", …)
    ejecucion     : timestamp de la ejecución (se genera en __main__ y se comparte)
    df_externo    : si se pasa, omite la ingesta y usa ese DataFrame directamente
    engine        : SQLAlchemy engine; si es None y guardar_oracle=True lo crea aquí
    guardar_oracle: si False omite todo guardado en Oracle (útil para pruebas locales)

    Retorna
    -------
    (lista_metricas, resultado_modelo)
    """
    log.info(f"\n{'='*60}")
    log.info(f"EJECUCIÓN: {sufijo}")
    log.info(f"{'='*60}")

    metricas       = []
    resultado_mod  = None

    # ── Etapa 1: Ingesta ──────────────────────────────────────────────────────
    if df_externo is None:
        df, m = medir_etapa("Ingesta", ingestar, ORIGEN, DESTINO)
        metricas.append(m)
        if df is None:
            log.error("Ingesta fallida, abortando pipeline.")
            return metricas, None
    else:
        df = df_externo
        log.info(f"[Ingesta] omitida — usando DataFrame externo ({len(df)} filas)")

    # ── Etapa 2: Limpieza ─────────────────────────────────────────────────────
    df_limpio, m = medir_etapa("Limpieza", limpiar, df, c_e)
    metricas.append(m)
    if df_limpio is None:
        log.error("Limpieza fallida, abortando pipeline.")
        return metricas, None

    # ── Etapa 3: Validación ───────────────────────────────────────────────────
    resultado_val, m = medir_etapa("Validación", validar, df_limpio)
    metricas.append(m)
    df_validos, df_errores = (
        resultado_val if resultado_val else (pd.DataFrame(), pd.DataFrame())
    )
    log.info(f"Válidos: {len(df_validos)} | Erróneos: {len(df_errores)}")

    # ── Etapa 4a: Guardado en disco (CSV) ─────────────────────────────────────
    _, m = medir_etapa("Guardado CSV", guardar_csv, df_validos, df_errores)
    metricas.append(m)

    # ── Etapa 5: Modelado ─────────────────────────────────────────────────────
    if len(df_validos) > 10:
        resultado_mod, m = medir_etapa("Modelado", modelar, df_validos)
        metricas.append(m)
    else:
        log.warning("Muy pocos registros válidos para modelar, etapa omitida.")

    # ── Etapa 4b: Guardado en Oracle ──────────────────────────────────────────
    if guardar_oracle and resultado_mod is not None:
        if engine is None:
            engine = get_engine()

        def _oracle_save():
            guardar_en_oracle(
                engine=engine,
                df_validos=df_validos,
                df_errores=df_errores,
                resultado_modelo=resultado_mod,
                lista_metricas=metricas,
                condicion=condicion,
                ejecucion=ejecucion,
            )

        _, m = medir_etapa("Guardado Oracle", _oracle_save)
        metricas.append(m)
    elif guardar_oracle and resultado_mod is None:
        log.warning("Guardado Oracle omitido — no hay resultado de modelo disponible.")

    return metricas, resultado_mod


# ── Ejecución principal ───────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("======== INICIO DEL PIPELINE ========")

    # Timestamp compartido por ambas ejecuciones para poder compararlas en Oracle
    ejecucion = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Conectar y crear tablas una sola vez
    engine = get_engine()
    crear_tablas(engine)

    # ── Condición 1: Dataset completo ─────────────────────────────────────────
    metricas_completo, _ = ejecutar_pipeline(
        sufijo        = "Dataset completo",
        condicion     = "completo",
        ejecucion     = ejecucion,
        engine        = engine,
        guardar_oracle= True,
    )

    # ── Condición 2: Dataset al 50% ───────────────────────────────────────────
    df_original = pd.read_csv(ORIGEN, encoding="utf-8-sig")
    df_mitad    = df_original.sample(frac=0.5, random_state=42).reset_index(drop=True)

    metricas_mitad, _ = ejecutar_pipeline(
        sufijo        = "Dataset 50%",
        condicion     = "50pct",
        ejecucion     = ejecucion,
        df_externo    = df_mitad,
        engine        = engine,
        guardar_oracle= True,
    )

    # ── Resumen comparativo en consola y CSV ──────────────────────────────────
    resumen_completo = resumen_metricas(metricas_completo)
    resumen_mitad    = resumen_metricas(metricas_mitad)

    resumen_completo.insert(0, "Condición", "Completo")
    resumen_mitad.insert(0, "Condición",   "50%")

    comparacion = pd.concat([resumen_completo, resumen_mitad], ignore_index=True)

    print("\n" + "=" * 80)
    print("RESUMEN COMPARATIVO DE RENDIMIENTO")
    print("=" * 80)
    print(comparacion.to_string(index=False))

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    comparacion.to_csv(
        "data/processed/rendimiento_comparativo.csv",
        index=False,
        encoding="utf-8-sig",
    )
    log.info("Métricas guardadas en data/processed/rendimiento_comparativo.csv")
    log.info("======== PIPELINE FINALIZADO ========")
