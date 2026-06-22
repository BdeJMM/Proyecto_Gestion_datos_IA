import logging
import time
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class MetricasEtapa:
    etapa: str
    latencia_seg: float
    filas_entrada: Optional[int]
    filas_salida: Optional[int]
    exito: bool
    error: Optional[str] = None


def _contar_filas(obj: Any) -> Optional[int]:
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return len(obj)
    if isinstance(obj, tuple):
        total = 0
        encontrado = False
        for item in obj:
            if isinstance(item, pd.DataFrame):
                total += len(item)
                encontrado = True
        return total if encontrado else None
    return None


def medir_etapa(nombre: str, func, *args, **kwargs):
    filas_entrada = None
    for a in args:
        filas_entrada = _contar_filas(a)
        if filas_entrada is not None:
            break

    log.info(f">> Iniciando etapa: {nombre}")
    t0 = time.time()
    try:
        resultado = func(*args, **kwargs)
        latencia = time.time() - t0
        filas_salida = _contar_filas(resultado)

        metrica = MetricasEtapa(
            etapa=nombre,
            latencia_seg=round(latencia, 4),
            filas_entrada=filas_entrada,
            filas_salida=filas_salida,
            exito=True,
        )
        log.info(f"OK {nombre} completada en {latencia:.3f}s "
                  f"(entrada={filas_entrada}, salida={filas_salida})")
        return resultado, metrica

    except Exception as e:
        latencia = time.time() - t0
        log.error(f"ERROR {nombre} fallo tras {latencia:.3f}s - {e}")
        metrica = MetricasEtapa(
            etapa=nombre,
            latencia_seg=round(latencia, 4),
            filas_entrada=filas_entrada,
            filas_salida=None,
            exito=False,
            error=str(e),
        )
        return None, metrica


def resumen_metricas(lista_metricas: list[MetricasEtapa]) -> pd.DataFrame:
    filas = []
    for m in lista_metricas:
        if m is None:
            continue
        completitud = None
        if m.filas_entrada and m.filas_entrada > 0 and m.filas_salida is not None:
            completitud = round(100 * m.filas_salida / m.filas_entrada, 2)

        filas.append({
            "Etapa": m.etapa,
            "Latencia (s)": m.latencia_seg,
            "Filas entrada": m.filas_entrada,
            "Filas salida": m.filas_salida,
            "Completitud (%)": completitud,
            "Éxito": "Sí" if m.exito else "No",
            "Error": m.error or "",
        })

    return pd.DataFrame(filas)
