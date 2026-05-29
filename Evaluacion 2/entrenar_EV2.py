"""
train.py
========
Entrena un modelo XGBoost sobre el dataset bank marketing.
Llamado por pipeline.py — nunca directamente por el usuario.

Recibe el DataFrame ya limpio y validado.
Retorna un TrainResult con el modelo, métricas y el df de predicciones.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score,
)
import xgboost as xgb

log = logging.getLogger(__name__)

# ── Columnas definidas por el Metadata ───────────────────────────────────────
COLUMNAS_NUMERICAS  = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
COLUMNAS_CATEGORICAS = ["job", "marital", "education", "default", "housing",
                        "loan", "contact", "month", "poutcome"]
COLUMNA_TARGET = "deposit"

# ── Hiperparámetros XGBoost (ajustados al benchmark real del dataset) ────────
PARAMS_XGBOOST = {
    "n_estimators":  300,
    "learning_rate": 0.05,
    "max_depth":     6,
    "subsample":     0.8,
    "colsample_bytree": 0.8,
    "random_state":  42,
    "eval_metric":   "logloss",
    "verbosity":     0,
}


# ── Resultado tipado ──────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    ok: bool
    modelo: Optional[xgb.XGBClassifier] = None
    encoders: dict = field(default_factory=dict)   # LabelEncoders para restaurar en servicio
    metricas: dict = field(default_factory=dict)
    df_predicciones: Optional[pd.DataFrame] = None
    errores: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


# ── Función principal ─────────────────────────────────────────────────────────

def entrenar(df: pd.DataFrame, ruta_modelo: str | Path = "models/model.pkl") -> TrainResult:
    """
    Preprocesa, entrena con validación cruzada y guarda el modelo.

    Parámetros
    ----------
    df          : DataFrame limpio y validado (salida de limpiar + validar).
    ruta_modelo : ruta donde se guardará el archivo .pkl.

    Retorna
    -------
    TrainResult con modelo, métricas, encoders y df de predicciones.
    """
    log.info("── Entrenamiento iniciado ──")

    # ── 1. Guardia de entrada ─────────────────────────────────────────────────
    if not isinstance(df, pd.DataFrame) or df.empty:
        msg = "df no es un DataFrame válido o está vacío"
        log.error(f"✗ {msg}")
        return TrainResult(ok=False, errores=[msg])

    cols_faltantes = [c for c in COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS + [COLUMNA_TARGET]
                      if c not in df.columns]
    if cols_faltantes:
        msg = f"Columnas faltantes en el DataFrame: {cols_faltantes}"
        log.error(f"✗ {msg}")
        return TrainResult(ok=False, errores=[msg])

    df = df.copy()

    # ── 2. Feature engineering específico para bank marketing ─────────────────
    # pdays == -1 significa "sin contacto previo" (según Metadata)
    # lo convertimos en flag binario para que el modelo lo entienda mejor
    df["pdays_contactado"] = (df["pdays"] != -1).astype(int)
    df["pdays"] = df["pdays"].replace(-1, 0)
    log.info("✓ Feature engineering: pdays_contactado creado")

    # ── 3. Encoding de categóricas ────────────────────────────────────────────
    encoders: dict[str, LabelEncoder] = {}
    try:
        for col in COLUMNAS_CATEGORICAS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            log.info(f"✓ '{col}' encoded ({len(le.classes_)} clases)")
    except Exception as exc:
        msg = f"Error en encoding: {exc}"
        log.error(f"✗ {msg}")
        return TrainResult(ok=False, errores=[msg])

    # ── 4. Separar X e y ─────────────────────────────────────────────────────
    feature_cols = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS + ["pdays_contactado"]
    X = df[feature_cols]
    y = (df[COLUMNA_TARGET].astype(str).str.lower().str.strip() == "yes").astype(int)

    log.info(f"✓ X shape: {X.shape} | y distribución: {y.value_counts().to_dict()}")

    # ── 5. Validación cruzada (5-fold estratificado) ──────────────────────────
    try:
        modelo_cv = xgb.XGBClassifier(**PARAMS_XGBOOST)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores_auc = cross_val_score(modelo_cv, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        log.info(f"✓ CV AUC-ROC: {scores_auc.mean():.4f} ± {scores_auc.std():.4f}")
    except Exception as exc:
        msg = f"Error en validación cruzada: {exc}"
        log.error(f"✗ {msg}")
        return TrainResult(ok=False, errores=[msg])

    # ── 6. Entrenamiento final sobre todo el dataset ──────────────────────────
    try:
        modelo_final = xgb.XGBClassifier(**PARAMS_XGBOOST)
        modelo_final.fit(X, y)
        log.info("✓ Modelo entrenado sobre dataset completo")
    except Exception as exc:
        msg = f"Error entrenando modelo final: {exc}"
        log.error(f"✗ {msg}")
        return TrainResult(ok=False, errores=[msg])

    # ── 7. Métricas sobre el dataset completo ────────────────────────────────
    try:
        y_pred  = modelo_final.predict(X)
        y_proba = modelo_final.predict_proba(X)[:, 1]

        metricas = {
            "auc_roc":   round(float(roc_auc_score(y, y_proba)), 4),
            "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y, y_pred, zero_division=0)), 4),
            "f1":        round(float(f1_score(y, y_pred, zero_division=0)), 4),
            "cv_auc_mean": round(float(scores_auc.mean()), 4),
            "cv_auc_std":  round(float(scores_auc.std()), 4),
            "n_filas":   int(len(df)),
            "n_features": int(X.shape[1]),
        }
        log.info(f"✓ AUC-ROC: {metricas['auc_roc']} | F1: {metricas['f1']} | "
                 f"Precision: {metricas['precision']} | Recall: {metricas['recall']}")
    except Exception as exc:
        msg = f"Error calculando métricas: {exc}"
        log.error(f"✗ {msg}")
        return TrainResult(ok=False, errores=[msg])

    # ── 8. DataFrame de predicciones ─────────────────────────────────────────
    try:
        df_pred = pd.DataFrame({
            "prob_yes":    np.round(y_proba, 4),
            "prediccion":  y_pred,
            "real":        y.values,
            "correcto":    (y_pred == y.values).astype(int),
            "run_ts":      pd.Timestamp.now(),
        })
        log.info(f"✓ df_predicciones generado: {df_pred.shape}")
    except Exception as exc:
        msg = f"Error generando df_predicciones: {exc}"
        log.error(f"✗ {msg}")
        return TrainResult(ok=False, errores=[msg])

    # ── 9. Guardar modelo en disco ────────────────────────────────────────────
    try:
        ruta_modelo = Path(ruta_modelo)
        ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta_modelo, "wb") as f:
            pickle.dump({"modelo": modelo_final, "encoders": encoders,
                         "features": feature_cols, "metricas": metricas}, f)
        log.info(f"✓ Modelo guardado en: {ruta_modelo}")
    except Exception as exc:
        # No es fatal — el resultado igual se retorna
        log.warning(f"⚠ No se pudo guardar el modelo en disco: {exc}")

    log.info("── Entrenamiento finalizado ──")
    return TrainResult(
        ok=True,
        modelo=modelo_final,
        encoders=encoders,
        metricas=metricas,
        df_predicciones=df_pred,
    )
