import logging
from sqlalchemy import text, inspect
from conexion import get_engine

log = logging.getLogger(__name__)

# ── Definición de tablas ──────────────────────────────────────────────────────
# 'default' → 'def_credit'  (DEFAULT es palabra reservada en SQL)
# 'day'     → 'day_contact' (DAY es palabra reservada en Oracle)

TABLAS = {
    "clientes_validos": """
        CREATE TABLE clientes_validos (
            id          NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            ejecucion   VARCHAR2(30),
            condicion   VARCHAR2(20),
            fecha_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            age         NUMBER,
            job         VARCHAR2(50),
            marital     VARCHAR2(20),
            education   VARCHAR2(20),
            def_credit  VARCHAR2(5),
            balance     NUMBER,
            housing     VARCHAR2(5),
            loan        VARCHAR2(5),
            contact     VARCHAR2(20),
            day_contact NUMBER,
            month       VARCHAR2(5),
            duration    NUMBER,
            campaign    NUMBER,
            pdays       NUMBER,
            previous    NUMBER,
            poutcome    VARCHAR2(20),
            deposit     VARCHAR2(5)
        )
    """,

    "registros_erroneos": """
        CREATE TABLE registros_erroneos (
            id            NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            ejecucion     VARCHAR2(30),
            condicion     VARCHAR2(20),
            fila_original NUMBER,
            motivo_error  VARCHAR2(1000),
            fecha_carga   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            age           NUMBER,
            job           VARCHAR2(50),
            marital       VARCHAR2(20),
            education     VARCHAR2(20),
            def_credit    VARCHAR2(5),
            balance       NUMBER,
            housing       VARCHAR2(5),
            loan          VARCHAR2(5),
            contact       VARCHAR2(20),
            day_contact   NUMBER,
            month         VARCHAR2(5),
            duration      NUMBER,
            campaign      NUMBER,
            pdays         NUMBER,
            previous      NUMBER,
            poutcome      VARCHAR2(20),
            deposit       VARCHAR2(5)
        )
    """,

    "metricas_modelo": """
        CREATE TABLE metricas_modelo (
            id          NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            ejecucion   VARCHAR2(30),
            condicion   VARCHAR2(20),
            modelo      VARCHAR2(50),
            accuracy    NUMBER(8,4),
            precision_m NUMBER(8,4),
            recall      NUMBER(8,4),
            f1_score    NUMBER(8,4),
            auc_roc     NUMBER(8,4),
            fecha_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    "rendimiento_pipeline": """
        CREATE TABLE rendimiento_pipeline (
            id            NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            ejecucion     VARCHAR2(30),
            condicion     VARCHAR2(20),
            etapa         VARCHAR2(30),
            latencia_seg  NUMBER(10,4),
            filas_entrada NUMBER,
            filas_salida  NUMBER,
            completitud   NUMBER(8,2),
            exito         VARCHAR2(5),
            error         VARCHAR2(500),
            fecha_carga   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
}


def crear_tablas(engine=None) -> None:
    """
    Verifica cuáles tablas no existen y las crea.
    Las que ya existen se omiten sin error.
    Acepta un engine externo para reusar la conexión del pipeline.
    """
    if engine is None:
        engine = get_engine()

    inspector = inspect(engine)
    tablas_existentes = [t.upper() for t in inspector.get_table_names()]
    log.info(f"Tablas existentes en el esquema: {tablas_existentes or 'ninguna'}")

    with engine.connect() as conn:
        for nombre, ddl in TABLAS.items():
            if nombre.upper() in tablas_existentes:
                log.info(f"Tabla ya existe, se omite: {nombre}")
            else:
                conn.execute(text(ddl))
                conn.commit()
                log.info(f"Tabla creada: {nombre}")

    log.info("Creación de tablas finalizada")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    crear_tablas()
