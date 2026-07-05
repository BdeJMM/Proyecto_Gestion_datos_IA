import logging
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

log = logging.getLogger(__name__)

COL_NUM = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
COL_CAT = ["job", "marital", "education", "default", "housing",
        "loan", "contact", "month", "poutcome"]
COL_OBJETIVO = "deposit"


def _preprocesador() -> ColumnTransformer:

    return ColumnTransformer([
        ("num", StandardScaler(), COL_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), COL_CAT),
    ])


def _evaluar(nombre: str, modelo, X_test, y_test) -> dict:

    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)[:, 1]
    y_test_bin = (y_test == "yes").astype(int)
    fpr, tpr, _ = roc_curve(y_test_bin, proba)

    metricas = {
        "modelo": nombre,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, pos_label="yes"),
        "recall": recall_score(y_test, pred, pos_label="yes"),
        "f1": f1_score(y_test, pred, pos_label="yes"),
        "auc_roc": roc_auc_score(y_test_bin, proba),
        "matriz_confusion": confusion_matrix(y_test, pred, labels=["no", "yes"]),
        "gini": 2 * roc_auc_score(y_test_bin, proba) - 1,
        # fpr/tpr se guardan para poder reconstruir la curva ROC más adelante
        # (en el dashboard, leyéndolos desde la base de datos) sin depender
        # de una imagen generada localmente.
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }

    log.info(f"-- {nombre} --")
    log.info(f"Accuracy:  {metricas['accuracy']:.4f}")
    log.info(f"Precision: {metricas['precision']:.4f}")
    log.info(f"Recall:    {metricas['recall']:.4f}")
    log.info(f"F1-score:  {metricas['f1']:.4f}")
    log.info(f"AUC-ROC:   {metricas['auc_roc']:.4f}")
    log.info(f"Gini:      {metricas['gini']:.4f}")
    log.info(f"Matriz de confusión [filas=real, cols=predicho] (orden: no, yes):\n{metricas['matriz_confusion']}")

    return metricas


def entrenar(df: pd.DataFrame, guardar_grafico: str | None = "reports/matriz_confusion.png"):
    log.info("-- Entrenamiento iniciado --")
    t0 = time.time()

    X = df[COL_NUM + COL_CAT]
    y = df[COL_OBJETIVO]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

    pipe_baseline = Pipeline([
        ("prep", _preprocesador()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe_baseline.fit(X_train, y_train)
    metricas_baseline = _evaluar("Regresión Logística (baseline)", pipe_baseline, X_test, y_test)

    pipe_xgb = Pipeline([
        ("prep", _preprocesador()),
        ("clf", XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        )),
    ])
    y_train_bin = (y_train == "yes").astype(int)
    pipe_xgb.fit(X_train, y_train_bin)
    class _Adaptador:
        def __init__(self, pipe):
            self.pipe = pipe
        def predict(self, X):
            return pd.Series(self.pipe.predict(X)).map({0: "no", 1: "yes"}).values
        def predict_proba(self, X):
            return self.pipe.predict_proba(X)

    metricas_xgb = _evaluar("XGBoost (modelo principal)", _Adaptador(pipe_xgb), X_test, y_test)

    if guardar_grafico:
        from pathlib import Path
        Path(guardar_grafico).parent.mkdir(parents=True, exist_ok=True)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=metricas_xgb["matriz_confusion"],
            display_labels=["no", "yes"],
        )
        disp.plot(cmap="Blues")
        disp.figure_.suptitle("Matriz de confusión — XGBoost")
        disp.figure_.savefig(guardar_grafico, dpi=150, bbox_inches="tight")
        log.info(f"Gráfico de matriz de confusión guardado en: {guardar_grafico}")

        # ── Curva ROC — compara Regresión Logística vs XGBoost ────────────
        ruta_roc = str(Path(guardar_grafico).parent / "curva_roc.png")
        disp_roc = RocCurveDisplay(
            fpr=metricas_baseline["fpr"], tpr=metricas_baseline["tpr"],
            roc_auc=metricas_baseline["auc_roc"],
        )
        disp_roc.plot(name="Regresión Logística")
        RocCurveDisplay(
            fpr=metricas_xgb["fpr"], tpr=metricas_xgb["tpr"],
            roc_auc=metricas_xgb["auc_roc"],
        ).plot(ax=disp_roc.ax_, name="XGBoost")
        disp_roc.ax_.legend(loc="lower right")
        disp_roc.figure_.suptitle("Curva ROC — comparación de modelos")
        disp_roc.figure_.savefig(ruta_roc, dpi=150, bbox_inches="tight")
        log.info(f"Gráfico de curva ROC guardado en: {ruta_roc}")

    log.info(f"Tiempo total de entrenamiento: {time.time() - t0:.2f} s")
    log.info("-- Entrenamiento finalizado --")

    return {
        "baseline": metricas_baseline,
        "xgboost": metricas_xgb,
        "modelo_xgb": pipe_xgb,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    from ingesta import ingestar, c_e
    from limpieza import limpiar
    from validacion import validar

    df_crudo = ingestar("02_bank.csv")
    df_limpio = limpiar(df_crudo, c_e)
    df_validos, df_errores = validar(df_limpio)

    resultado = entrenar(df_validos)

    print("\n=== RESUMEN COMPARATIVO ===")
    for clave in ("baseline", "xgboost"):
        m = resultado[clave]
        print(f"{m['modelo']:35s} | AUC={m['auc_roc']:.3f} | F1={m['f1']:.3f} | Acc={m['accuracy']:.3f}")