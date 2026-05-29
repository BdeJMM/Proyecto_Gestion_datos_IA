# ── Imagen base ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

LABEL maintainer="DuocUC ITY1101"
LABEL description="Pipeline DataOps — Bank Marketing XGBoost"

ENV DB_URL="sqlite:///data/bank.db" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (aprovecha caché de Docker)
COPY "Evaluacion 2/requirements.txt" .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todos los módulos del pipeline
COPY "Evaluacion 2/Ingesta_EV2.py"     .
COPY "Evaluacion 2/limpieza_EV2.py"    .
COPY "Evaluacion 2/validacion_EV2.py"  .
COPY "Evaluacion 2/entrenar_EV2.py"    .
COPY "Evaluacion 2/exportar_EV2.py"    .
COPY "Evaluacion 2/pipeline_EV2.py"    .
COPY "Evaluacion 2/02_bank.csv"        data/raw/bank.csv

# Carpetas de salida
RUN mkdir -p data/raw data/processed data/backup models reports

ENTRYPOINT ["python", "pipeline_EV2.py"]
CMD ["--fuente", "data/raw/bank.csv", \
     "--destino", "data/processed/bank_work.csv", \
     "--respaldo", "data/backup/bank_backup.csv", \
     "--modelo", "models/model.pkl"]
