import logging
from sqlalchemy import text, inspect
from conexion import get_engine

log = logging.getLogger(__name__)

TABLAS = {
    "clientes_validos": """
        CREATE TABLE clientes_validos (
            id          SERIAL PRIMARY KEY,
            ejecucion   VARCHAR(30),
            condicion   VARCHAR(20),
            fecha_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            age         NUMERIC,
            job         VARCHAR(50),
            marital     VARCHAR(20),
            education   VARCHAR(20),
            def_credit  VARCHAR(5),
            balance     NUMERIC,
            housing     VARCHAR(5),
            loan        VARCHAR(5),
            contact     VARCHAR(20),
            day_contact NUMERIC,
            month       VARCHAR(5),
            duration    NUMERIC,
            campaign    NUMERIC,
            pdays       NUMERIC,
            previous    NUMERIC,
            poutcome    VARCHAR(20),
            deposit     VARCHAR(5)
        )
    """,

    "registros_erroneos": """
        CREATE TABLE registros_erroneos (
            id            SERIAL PRIMARY KEY,
            ejecucion     VARCHAR(30),
            condicion     VARCHAR(20),
            fila_original NUMERIC,
            motivo_error  VARCHAR(1000),
            fecha_carga   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            age           NUMERIC,
            job           VARCHAR(50),
            marital       VARCHAR(20),
            education     VARCHAR(20),
            def_credit    VARCHAR(5),
            balance       NUMERIC,
            housing       VARCHAR(5),
            loan          VARCHAR(5),
            contact       VARCHAR(20),
            day_contact   NUMERIC,
            month         VARCHAR(5),
            duration      NUMERIC,
            campaign      NUMERIC,
            pdays         NUMERIC,
            previous      NUMERIC,
            poutcome      VARCHAR(20),
            deposit       VARCHAR(5)
        )
    """,

    "metricas_modelo": """
        CREATE TABLE metricas_modelo (
            id          SERIAL PRIMARY KEY,
            ejecucion   VARCHAR(30),
            condicion   VARCHAR(20),
            modelo      VARCHAR(50),
            accuracy    NUMERIC(8,4),
            precision_m NUMERIC(8,4),
            recall      NUMERIC(8,4),
            f1_score    NUMERIC(8,4),
            auc_roc     NUMERIC(8,4),
            gini        NUMERIC(8,4),
            fecha_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    "rendimiento_pipeline": """
        CREATE TABLE rendimiento_pipeline (
            id            SERIAL PRIMARY KEY,
            ejecucion     VARCHAR(30),
            condicion     VARCHAR(20),
            etapa         VARCHAR(30),
            latencia_seg  NUMERIC(10,4),
            filas_entrada NUMERIC,
            filas_salida  NUMERIC,
            completitud   NUMERIC(8,2),
            exito         VARCHAR(5),
            error         VARCHAR(500),
            fecha_carga   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    "matriz_confusion": """
        CREATE TABLE matriz_confusion (
            id          SERIAL PRIMARY KEY,
            ejecucion   VARCHAR(30),
            condicion   VARCHAR(20),
            modelo      VARCHAR(50),
            tn          NUMERIC,
            fp          NUMERIC,
            fn          NUMERIC,
            tp          NUMERIC,
            fecha_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    "curva_roc": """
        CREATE TABLE curva_roc (
            id          SERIAL PRIMARY KEY,
            ejecucion   VARCHAR(30),
            condicion   VARCHAR(20),
            modelo      VARCHAR(50),
            fpr         TEXT,
            tpr         TEXT,
            auc_roc     NUMERIC(8,4),
            fecha_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
}


def crear_tablas(engine=None) -> None:
    if engine is None:
        engine = get_engine()

    inspector = inspect(engine)
    tablas_existentes = [t.lower() for t in inspector.get_table_names()]
    log.info(f"Tablas existentes: {tablas_existentes or 'ninguna'}")

    with engine.connect() as conn:
        for nombre, ddl in TABLAS.items():
            if nombre.lower() in tablas_existentes:
                log.info(f"Tabla ya existe, se omite: {nombre}")
            else:
                conn.execute(text(ddl))
                conn.commit()
                log.info(f"Tabla creada: {nombre}")

        # ── Migración: agregar columna 'gini' si la tabla ya existía sin ella ──
        if "metricas_modelo" in tablas_existentes:
            columnas = [c["name"].lower() for c in inspector.get_columns("metricas_modelo")]
            if "gini" not in columnas:
                conn.execute(text(
                    "ALTER TABLE metricas_modelo ADD COLUMN gini NUMERIC(8,4)"
                ))
                conn.commit()
                log.info("Columna 'gini' agregada a metricas_modelo (migración automática)")
            else:
                log.info("Columna 'gini' ya existe en metricas_modelo")

    log.info("Creación de tablas finalizada")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    crear_tablas()