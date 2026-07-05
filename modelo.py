import logging
import pandas as pd

from entrenar import entrenar

log = logging.getLogger(__name__)


def modelar(df: pd.DataFrame) -> dict:
    
    log.info("-- Modelado iniciado --")
    resultado = entrenar(df)

    m_base = resultado["baseline"]
    m_xgb = resultado["xgboost"]
    log.info(
        f"Comparación — Baseline AUC={m_base['auc_roc']:.3f} | "
        f"XGBoost AUC={m_xgb['auc_roc']:.3f}"
    )
    log.info("-- Modelado finalizado --")

    return resultado
