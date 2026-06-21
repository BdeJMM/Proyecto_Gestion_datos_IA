import logging
import pandas as pd
from conexion_oracle import get_engine
 
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)
 
 
def ver_evolución_auc() -> pd.DataFrame:
    """Muestra cómo evoluciona el AUC del modelo en cada ejecución."""
    engine = get_engine()
    df = pd.read_sql("""
        SELECT ejecucion,
               condicion,
               modelo,
               auc_roc,
               f1_score,
               accuracy,
               fecha_carga
        FROM metricas_modelo
        ORDER BY fecha_carga DESC
    """, engine)
    return df
 
 
def ver_depositos_por_trabajo() -> pd.DataFrame:
    """Cuántos clientes depositan según su trabajo."""
    engine = get_engine()
    df = pd.read_sql("""
        SELECT job,
               COUNT(*)                                          AS total,
               SUM(CASE WHEN deposit = 'yes' THEN 1 ELSE 0 END) AS depositaron,
               ROUND(
                   SUM(CASE WHEN deposit = 'yes' THEN 1 ELSE 0 END)
                   * 100.0 / COUNT(*), 2
               )                                                 AS pct_deposito
        FROM clientes_validos
        GROUP BY job
        ORDER BY total DESC
    """, engine)
    return df
 
 
def ver_rendimiento_pipeline() -> pd.DataFrame:
    """Latencia y completitud de cada etapa por ejecución."""
    engine = get_engine()
    df = pd.read_sql("""
        SELECT ejecucion,
               condicion,
               etapa,
               latencia_seg,
               filas_entrada,
               filas_salida,
               completitud,
               exito,
               fecha_carga
        FROM rendimiento_pipeline
        ORDER BY fecha_carga DESC, etapa
    """, engine)
    return df
 
 
def ver_errores_recientes(n: int = 20) -> pd.DataFrame:
    """Últimos N registros erróneos con su motivo."""
    engine = get_engine()
    df = pd.read_sql(f"""
        SELECT ejecucion,
               condicion,
               fila_original,
               motivo_error,
               age,
               job,
               deposit,
               fecha_carga
        FROM registros_erroneos
        ORDER BY fecha_carga DESC
        FETCH FIRST {n} ROWS ONLY
    """, engine)
    return df
 
 
def ver_resumen_ejecuciones() -> pd.DataFrame:
    """Resumen de cuántos registros válidos y erróneos por ejecución."""
    engine = get_engine()
    df = pd.read_sql("""
        SELECT v.ejecucion,
               v.condicion,
               v.total_validos,
               NVL(e.total_errores, 0) AS total_errores,
               v.fecha_carga
        FROM (
            SELECT ejecucion, condicion,
                   COUNT(*) AS total_validos,
                   MAX(fecha_carga) AS fecha_carga
            FROM clientes_validos
            GROUP BY ejecucion, condicion
        ) v
        LEFT JOIN (
            SELECT ejecucion, condicion,
                   COUNT(*) AS total_errores
            FROM registros_erroneos
            GROUP BY ejecucion, condicion
        ) e ON v.ejecucion = e.ejecucion AND v.condicion = e.condicion
        ORDER BY v.fecha_carga DESC
    """, engine)
    return df
 
 
# ── Ejecución directa ────────────────────────────────────────────────────────
if __name__ == "__main__":
 
    print("\n" + "="*60)
    print("RESUMEN DE EJECUCIONES")
    print("="*60)
    print(ver_resumen_ejecuciones().to_string(index=False))
 
    print("\n" + "="*60)
    print("EVOLUCIÓN DEL AUC POR MODELO")
    print("="*60)
    print(ver_evolución_auc().to_string(index=False))
 
    print("\n" + "="*60)
    print("DEPÓSITOS POR TIPO DE TRABAJO")
    print("="*60)
    print(ver_depositos_por_trabajo().to_string(index=False))
 
    print("\n" + "="*60)
    print("RENDIMIENTO DEL PIPELINE (últimas ejecuciones)")
    print("="*60)
    print(ver_rendimiento_pipeline().to_string(index=False))
 
    print("\n" + "="*60)
    print("ÚLTIMOS 20 REGISTROS ERRÓNEOS")
    print("="*60)
    print(ver_errores_recientes(20).to_string(index=False))