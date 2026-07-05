# Gestión de Datos IA — Pipeline de Depósitos Bancarios

Pipeline de datos que ingiere, limpia, valida y modela un dataset de clientes bancarios para predecir si suscribirán un depósito a plazo, guardando resultados y métricas en una base de datos **PostgreSQL (Neon)**.

## Estructura del proyecto

```
├── 02_bank.csv          # Dataset de entrada
├── 01_Metadata.txt       # Descripción de las columnas del dataset
├── ingesta.py            # Etapa 1: carga y validación estructural del CSV
├── limpieza.py           # Etapa 2: imputación de nulos, normalización de tipos
├── validacion.py         # Etapa 3: reglas semánticas y de coherencia de negocio
├── guardado.py           # Etapa 4: persistencia en CSV y en la base de datos
├── modelo.py / entrenar.py  # Etapa 5: entrenamiento (Regresión Logística + XGBoost)
├── rendimiento.py        # Medición de latencia/completitud de cada etapa
├── tablas.py              # Creación de tablas en la base de datos
├── conexion.py            # Conexión a la base de datos vía SQLAlchemy
├── consultas.py            # Consultas de análisis sobre los datos guardados
├── pipeline.py             # Orquestador — corre el pipeline completo
├── dashboard.py             # Dashboard técnico (Streamlit)
├── pages/                   # Páginas adicionales del dashboard (ej. vista para analistas)
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

## Requisitos

- Python 3.11+
- Una base de datos PostgreSQL — este proyecto está configurado para **[Neon](https://neon.tech)** (plan gratuito disponible)
- Docker y Docker Compose (opcional, para correr en contenedor)

## Configuración

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Crea un archivo `.env` en la raíz del proyecto con la cadena de conexión de tu base en Neon:
   ```
   DATABASE_URL=postgresql+psycopg2://usuario:password@ep-xxxx-xxxx.region.aws.neon.tech/nombre_bd?sslmode=require
   ```
   Puedes obtener este string desde el dashboard de Neon, en **Connect to your database**. Nota el prefijo `+psycopg2` (necesario para que SQLAlchemy sepa qué driver usar) y el parámetro `sslmode=require` (Neon exige SSL en todas las conexiones).

   **Nunca subas `.env` a git** — ya está excluido en `.gitignore`.

## Uso

### Correr el pipeline completo

```bash
python pipeline.py
```

Esto ejecuta dos corridas (dataset completo y al 50%), y para cada una:
1. Ingiere `02_bank.csv`
2. Limpia y valida los datos
3. Guarda los registros válidos/erróneos en `data/` (CSV) y en la base de datos
4. Entrena Regresión Logística y XGBoost, y guarda las métricas
5. Registra el rendimiento de cada etapa

Los logs de cada corrida quedan en `logs/`, y un resumen comparativo en `data/processed/rendimiento_comparativo.csv`.

### Ver el dashboard técnico

```bash
streamlit run dashboard.py
```

Si existe una carpeta `pages/` junto a `dashboard.py`, Streamlit agrega automáticamente esas páginas al menú lateral de navegación (por ejemplo, una vista simplificada de estadísticas del modelo pensada para analistas).

### Consultar los datos por línea de comandos

```bash
python consultas.py
```

Imprime en consola resúmenes de ejecuciones, evolución del AUC, depósitos por tipo de trabajo, rendimiento del pipeline y errores recientes.

### Correr con Docker

```bash
docker compose up --build
```

Esto construye la imagen, instala dependencias y corre `pipeline.py` dentro del contenedor. Los directorios `data/`, `logs/` y `reports/` se montan como volúmenes para persistir los resultados fuera del contenedor. Asegúrate de tener tu `.env` en la raíz del proyecto antes de levantar el servicio.

## Modelo

Se entrenan y comparan dos modelos sobre la variable objetivo `deposit`:
- **Regresión Logística** (baseline)
- **XGBoost** (modelo principal)

Métricas registradas: Accuracy, Precisión, Recall, F1-Score y AUC-ROC. La matriz de confusión del modelo XGBoost se guarda como imagen en `reports/matriz_confusion.png`.

## Seguridad

- La base de datos de Neon requiere SSL en todas las conexiones (`sslmode=require`).
- Si vas a conectar herramientas externas de BI/análisis (Looker Studio, Power BI, Grafana, etc.), usa un **rol de solo lectura** dedicado en Postgres — nunca el rol dueño de la base que usa el pipeline para escribir. Ver la sección de gestión de roles en la [documentación de Neon](https://neon.com/docs/manage/database-access).
- Las credenciales viven únicamente en `.env` (excluido de git) o en variables de entorno del entorno de despliegue — nunca hardcodeadas en el código.
