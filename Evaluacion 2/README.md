# 🛡️ Detección de Fraude en Tarjetas de Crédito — Pipeline DataOps

> **ITY1101 — Gestión de Datos para IA | Evaluación Parcial N°2 | DuocUC 2025**

Pipeline DataOps completo para detección de transacciones fraudulentas con tarjetas de crédito, implementado con Python, TensorFlow y Google Colab.

---

## 📋 Descripción del Proyecto

Este proyecto implementa un pipeline de datos de 4 etapas para clasificar transacciones como legítimas o fraudulentas (variable `is_fraud`), basado en el **Caso 3 — Credit Card Fraud Detection**.

El dataset sintético generado contiene **19.000 transacciones** (~5% fraude), replicando patrones reales: montos elevados, horarios nocturnos y mayor distancia geográfica entre titular y comercio.

---

## 🗂️ Estructura del Repositorio

```
fraude-deteccion-dataops/
│
├── notebooks/
│   ├── Creacion_de_datos_EV2.ipynb       # Etapa 1: Ingesta y generación de datos
│   ├── Limpieza_de_datos_EV2.ipynb       # Etapa 2: Limpieza y transformación
│   ├── Validacion_de_datos_EV2.ipynb     # Etapa 3: Validación estructural y semántica
│   └── Entrenamiento_de_IA_EV2.ipynb     # Etapa 4: Entrenamiento del modelo
│
├── Dockerfile                             # Contenedor para ejecución reproducible
├── requirements.txt                       # Dependencias del proyecto
└── README.md
```

---

## ⚙️ Pipeline DataOps

### Etapa 1 — Ingesta (`Creacion_de_datos_EV2.ipynb`)
- Genera 9.000 transacciones legítimas y 500 fraudulentas de forma sintética
- Variables: identidad del titular, coordenadas geográficas, monto, categoría, comercio
- Salida: `datos_generados.csv`

### Etapa 2 — Limpieza y Transformación (`Limpieza_de_datos_EV2.ipynb`)
- Detección de nulos y eliminación de duplicados
- Feature engineering: `hour`, `is_night`, `geo_distance`, `amt_zscore`
- Codificación de categóricas con `LabelEncoder`
- Salida: `datos_limpios.csv` con 11 features finales

### Etapa 3 — Validación (`Validacion_de_datos_EV2.ipynb`)
- **Estructural:** nulos, rangos válidos, flags binarios
- **Semántica:** fraudes con mayor monto, mayor distancia y mayor proporción nocturna
- El pipeline se detiene con `ValueError` si alguna validación falla

### Etapa 4 — Entrenamiento (`Entrenamiento_de_IA_EV2.ipynb`)
- Split 80/20 estratificado + StandardScaler + SMOTE para balanceo
- Red neuronal densa: `Dense(64) → BatchNorm → Dropout → Dense(32) → Sigmoid`
- Métricas: ROC-AUC, F1-Score, Precision, Recall
- Salida: modelo `.h5` + scaler `.pkl`

---

## 🚀 Cómo Ejecutar

### Opción A — Google Colab
1. Sube los notebooks a tu Google Drive en la ruta:
   `MyDrive/Colab Notebooks/Gestion_Datos_IA_EV2/`
2. Ejecuta los notebooks en orden (Etapa 1 → 4)

### Opción B — Docker
```bash
# Construir imagen
docker build -t fraude-dataops .

# Ejecutar pipeline completo
docker run --rm fraude-dataops
```

---

## 🧰 Tecnologías Utilizadas

| Herramienta | Uso |
|---|---|
| Python 3.10 | Lenguaje principal |
| pandas / numpy | Manipulación de datos |
| scikit-learn | Preprocesamiento y métricas |
| imbalanced-learn | SMOTE para balanceo de clases |
| TensorFlow / Keras | Modelo de red neuronal |
| Google Colab + Drive | Entorno de ejecución y almacenamiento |
| Docker | Contenedorización del pipeline |
| GitHub Projects | Seguimiento del proyecto (Kanban) |

---

## 👥 Integrantes

| Nombre | Rol |
|---|---|
| [Nombre Apellido] | Data Engineer — Etapa 1 |
| [Nombre Apellido] | Data Analyst — Etapas 2 y 3 |
| [Nombre Apellido] | ML Engineer — Etapa 4 |

---

## 📄 Licencia

Proyecto académico — DuocUC 2025. Solo para fines educativos.
