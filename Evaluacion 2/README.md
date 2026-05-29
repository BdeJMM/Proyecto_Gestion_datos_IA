# 🏦 Bank Marketing — Pipeline DataOps

> **ITY1101 — Gestión de Datos para IA | Evaluación Parcial N°2 | DuocUC 2025**

Pipeline DataOps completo para predecir si un cliente suscribirá un depósito a plazo fijo, implementado en Python con XGBoost y exportación automática a base de datos SQL.

---

## 📋 Descripción del Proyecto

Este proyecto implementa un pipeline de datos de 5 etapas para clasificar clientes de un banco según su probabilidad de suscribir un depósito a plazo fijo (variable `deposit`), basado en el dataset **Bank Marketing**.

El dataset contiene información de campañas de marketing telefónico, incluyendo datos demográficos, financieros y del historial de contacto de cada cliente.

---

## 🗂️ Estructura del Repositorio

```
bank-marketing-dataops/
│
├── 02_bank.csv                  # Dataset original
├── 01_Metadata.txt              # Descripción de variables
│
├── Ingesta_EV2.py               # Etapa 1: Copia y carga del CSV
├── limpieza_EV2.py              # Etapa 2: Limpieza y normalización
├── validacion_EV2.py            # Etapa 3: Validación estructural y semántica
├── entrenar_EV2.py              # Etapa 4: Entrenamiento XGBoost
├── exportar_EV2.py              # Etapa 5: Exportación a base de datos SQL
├── pipeline_EV2.py              # Orquestador principal
│
├── data/
│   ├── raw/                     # CSV original (copiado por ingesta)
│   ├── processed/               # CSV de trabajo
│   └── backup/                  # Respaldo automático
│
├── models/
│   └── model.pkl                # Modelo entrenado (generado al ejecutar)
│
├── reports/
│   └── pipeline.log             # Log detallado de cada ejecución
│
├── Dockerfile                   # Contenedor para ejecución reproducible
├── requirements.txt             # Dependencias del proyecto
└── README.md
```

---

## ⚙️ Pipeline DataOps — 5 Etapas

### Etapa 1 — Ingesta (`Ingesta_EV2.py`)
- Verifica que el CSV de origen exista
- Copia el archivo a `data/processed/` y genera un respaldo en `data/backup/`
- Carga el DataFrame con `pandas` y valida que no esté vacío
- Salida: DataFrame con los datos crudos

### Etapa 2 — Limpieza (`limpieza_EV2.py`)
- Detecta automáticamente columnas numéricas, de texto y de fecha
- Imputa nulos en numéricas con la mediana
- Normaliza texto (strip + title case) y elimina filas con valores vacíos
- Elimina duplicados
- Salida: DataFrame limpio

### Etapa 3 — Validación (`validacion_EV2.py`)
- Verifica que no haya nulos ni duplicados residuales
- Controla valores infinitos en columnas numéricas
- Detecta strings vacíos o "nan" como texto
- El pipeline se detiene si alguna validación falla
- Salida: `True` (aprobado) / `False` (fallido)

### Etapa 4 — Entrenamiento (`entrenar_EV2.py`)
- Feature engineering: indicador `pdays_contactado` (contacto previo sí/no)
- Encoding de variables categóricas con `LabelEncoder`
- Validación cruzada estratificada (5-fold) con AUC-ROC
- Entrenamiento final de XGBoost sobre el dataset completo
- Guarda el modelo en `models/model.pkl`
- Salida: modelo, métricas y DataFrame de predicciones

### Etapa 5 — Exportación SQL (`exportar_EV2.py`)
- Escribe tres tablas en la base de datos:
  - `bank_clientes` — datos limpios (reemplaza en cada run)
  - `bank_predicciones` — probabilidades y predicciones del modelo
  - `bank_runs` — historial acumulativo de métricas por ejecución
- Compatible con SQLite, PostgreSQL (Amazon RDS) y Oracle Cloud
- Salida: `ExportResult` con conteo de filas exportadas

---

## 🚀 Cómo Ejecutar

### Requisitos previos
- Python 3.10+
- Las dependencias listadas en `requirements.txt`

### Instalación local

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd bank-marketing-dataops

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el pipeline completo
python pipeline_EV2.py --fuente 02_bank.csv
```

### Opciones de línea de comandos

```bash
python pipeline_EV2.py \
  --fuente   data/raw/bank.csv \        # CSV de entrada
  --destino  data/processed/bank.csv \  # Copia de trabajo
  --respaldo data/backup/bank.csv \     # Respaldo
  --modelo   models/model.pkl \         # Ruta del modelo
  --db-url   sqlite:///data/bank.db     # Base de datos (opcional)
```

**Ejemplos de `--db-url`:**

| Motor | Formato |
|---|---|
| SQLite (local, por defecto) | `sqlite:///data/bank.db` |
| PostgreSQL / Amazon RDS | `postgresql+psycopg2://user:pass@host:5432/bank` |
| Oracle Cloud | `oracle+cx_oracle://user:pass@host:1521/ORCL` |
| MySQL / Amazon RDS MySQL | `mysql+pymysql://user:pass@host:3306/bank` |

Si no se pasa `--db-url`, el pipeline busca la variable de entorno `DB_URL`. Si tampoco existe, usa SQLite local como fallback.

### Docker

```bash
# Construir imagen
docker build -t bank-dataops .

# Ejecutar con SQLite local
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  bank-dataops

# Ejecutar con base de datos externa
docker run --rm \
  -e DB_URL="postgresql+psycopg2://user:pass@host:5432/bank" \
  -v $(pwd)/data:/app/data \
  bank-dataops
```

---

## 🧰 Tecnologías Utilizadas

| Herramienta | Uso |
|---|---|
| Python 3.10 | Lenguaje principal |
| pandas / numpy | Manipulación de datos |
| scikit-learn | Encoding, validación cruzada y métricas |
| XGBoost | Modelo de clasificación |
| SQLAlchemy | Exportación a base de datos |
| Docker | Contenedorización del pipeline |
| GitHub Actions | CI/CD automatizado |

---

## 📊 Métricas del Modelo

El modelo XGBoost se evalúa con validación cruzada estratificada (5-fold). Las métricas se registran automáticamente en la tabla `bank_runs` de la base de datos y en `reports/pipeline.log` después de cada ejecución.

| Métrica | Descripción |
|---|---|
| AUC-ROC (CV) | Área bajo la curva ROC — promedio de 5 folds |
| AUC-ROC (total) | AUC sobre el dataset completo |
| F1-Score | Balance entre precisión y recall |
| Precision | Exactitud de las predicciones positivas |
| Recall | Cobertura de los positivos reales |

---

## 👥 Integrantes

| Nombre | Rol |
|---|---|
| [Nombre Apellido] | Data Engineer — Etapas 1 y 2 |
| [Nombre Apellido] | Data Analyst — Etapa 3 |
| [Nombre Apellido] | ML Engineer — Etapas 4 y 5 |

---

## 📄 Licencia

Proyecto académico — DuocUC 2025. Solo para fines educativos.
