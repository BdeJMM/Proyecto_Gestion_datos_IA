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

    metricas = {
        "modelo": nombre,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, pos_label="yes"),
        "recall": recall_score(y_test, pred, pos_label="yes"),
        "f1": f1_score(y_test, pred, pos_label="yes"),
        "auc_roc": roc_auc_score((y_test == "yes").astype(int), proba),
        "matriz_confusion": confusion_matrix(y_test, pred, labels=["no", "yes"]),
        "gini": 2 * roc_auc_score((y_test == "yes").astype(int), proba) - 1,
        # _y_test_bin y _proba se guardan aparte (no van a la base de datos),
        # se usan más abajo para dibujar la curva ROC comparando ambos modelos.
        "_y_test_bin": (y_test == "yes").astype(int),
        "_proba": proba,
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

    # ── Preprocesador compartido — se ajusta UNA sola vez, no una por modelo ──
    prep = _preprocesador()
    X_train_t = prep.fit_transform(X_train, y_train)
    X_test_t = prep.transform(X_test)

    modelo_baseline = LogisticRegression(max_iter=1000, random_state=42)
    modelo_baseline.fit(X_train_t, y_train)
    # El Pipeline se arma con el preprocesador YA ajustado — llamar .predict()
    # sobre él no vuelve a ajustar nada, solo encadena transform() + predict().
    pipe_baseline = Pipeline([("prep", prep), ("clf", modelo_baseline)])
    metricas_baseline = _evaluar("Regresión Logística (baseline)", pipe_baseline, X_test, y_test)

    y_train_bin = (y_train == "yes").astype(int)

    # Se separa un pequeño set de validación DEL TRAIN (nunca del test) para
    # que el early stopping decida cuándo parar sin tocar los datos de prueba.
    X_tr_xgb, X_val_xgb, y_tr_xgb, y_val_xgb = train_test_split(
        X_train_t, y_train_bin, test_size=0.15, random_state=42, stratify=y_train_bin
    )

    modelo_xgb = XGBClassifier(
        n_estimators=1000,       # techo alto — el early stopping decide cuándo parar
        max_depth=5,
        learning_rate=0.05,      # más bajo + más rondas = aprendizaje más fino
        subsample=0.8,           # cada árbol ve 80% de las filas (regulariza + acelera)
        colsample_bytree=0.8,    # cada árbol ve 80% de las columnas (regulariza + acelera)
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",      # método de construcción de árboles más rápido
        n_jobs=-1,               # usa todos los núcleos disponibles
        early_stopping_rounds=30,
    )
    modelo_xgb.fit(
        X_tr_xgb, y_tr_xgb,
        eval_set=[(X_val_xgb, y_val_xgb)],
        verbose=False,
    )
    log.info(
        f"XGBoost — detenido en la ronda {modelo_xgb.best_iteration} "
        f"de {modelo_xgb.n_estimators} configuradas (early stopping)"
    )
    pipe_xgb = Pipeline([("prep", prep), ("clf", modelo_xgb)])
    class _Adaptador:
        def __init__(self, pipe):
            self.pipe = pipe
        def predict(self, X):
            return pd.Series(self.pipe.predict(X)).map({0: "no", 1: "yes"}).values
        def predict_proba(self, X):
            return self.pipe.predict_proba(X)

    metricas_xgb = _evaluar("XGBoost (modelo principal)", _Adaptador(pipe_xgb), X_test, y_test)

    # ── Curva ROC: se calcula siempre (no solo si se guarda la imagen), ──────
    # porque guardado.py necesita fpr/tpr para insertarlos en la base de datos.
    fpr_base, tpr_base, _ = roc_curve(
        metricas_baseline["_y_test_bin"], metricas_baseline["_proba"]
    )
    fpr_xgb, tpr_xgb, _ = roc_curve(
        metricas_xgb["_y_test_bin"], metricas_xgb["_proba"]
    )
    # Se guardan como listas (JSON-serializables) para la base de datos.
    # OJO: para graficar más abajo se usan fpr_base/tpr_base (arrays de numpy),
    # NUNCA estas listas — pasarle una lista de Python a RocCurveDisplay hace
    # que scikit-learn (>=1.7) la interprete como "múltiples curvas" en vez de
    # "una curva con muchos puntos", y truena con un error de longitudes.
    metricas_baseline["fpr"] = fpr_base.tolist()
    metricas_baseline["tpr"] = tpr_base.tolist()
    metricas_xgb["fpr"] = fpr_xgb.tolist()
    metricas_xgb["tpr"] = tpr_xgb.tolist()

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
        disp_roc = RocCurveDisplay(fpr=fpr_base, tpr=tpr_base, roc_auc=metricas_baseline["auc_roc"])
        disp_roc.plot(name="Regresión Logística")
        RocCurveDisplay(fpr=fpr_xgb, tpr=tpr_xgb, roc_auc=metricas_xgb["auc_roc"]).plot(
            ax=disp_roc.ax_, name="XGBoost"
        )
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