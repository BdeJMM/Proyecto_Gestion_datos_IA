# Pipeline de Procesamiento de Datos VR

Pipeline modular en Python para ingestar, limpiar, transformar, validar y guardar el dataset `data.csv` de experiencias de usuario en dispositivos de Realidad Virtual.

---

## Estructura del proyecto

```
├── data.csv                        # CSV de origen (fuente)
├── ingesta.py                      # Copia a raw/, hace respaldo y carga el CSV
├── limpieza.py                     # Limpieza genérica por columna
├── validacion.py                   # Validación post-limpieza
├── guardado.py                     # Guarda el CSV procesado
├── pipeline.py                     # Orquesta todas las etapas
├── data/
│   ├── raw/data.csv                # CSV copiado en la ingesta
│   ├── backup/data.csv             # Respaldo automático del original
│   └── processed/data_clean.csv   # Dataset final limpio
└── logs/
    └── pipeline.log                # Log completo de cada ejecución
```

---

## Dataset

**Archivo fuente:** `data.csv`  
**Filas:** 1 000 | **Columnas originales:** 7

| Columna | Tipo | Descripción |
|---|---|---|
| UserID | int | Identificador único del usuario |
| Age | int | Edad del usuario (18–60) |
| Gender | str | Género (Male / Female / Other) |
| VRHeadset | str | Dispositivo VR utilizado |
| Duration | float | Duración de la sesión en minutos |
| MotionSickness | int | Nivel de mareo (1–10) |
| ImmersionLevel | int | Nivel de inmersión percibido (1–5) |

---

## Etapas del pipeline

### 1. Ingesta (`ingesta.py`)
- Valida que el archivo de origen exista
- Copia el CSV a `data/raw/`
- Genera un respaldo en `data/backup/`
- Carga el DataFrame sin modificarlo

### 2. Limpieza (`limpieza.py`)
Detecta automáticamente el tipo de cada columna y aplica la limpieza correspondiente:
- **Numéricas:** convierte a numérico, imputa nulos con la mediana
- **Texto:** strip + title case, elimina filas vacías o nulas
- **Fechas:** convierte a datetime, elimina fechas inválidas
- Elimina duplicados por `UserID`

### 3. Transformación (`pipeline.py`)
- Redondea columnas float a 2 decimales
- Crea columnas derivadas:

| Columna nueva | Descripción |
|---|---|
| `AgeGroup` | Segmento etario: Young / Middle / Senior |
| `DurationCategory` | Duración de sesión: Short / Medium / Long |
| `ComfortScore` | Índice de confort neto 0–10 (inmersión vs mareo) |

### 4. Validación (`validacion.py`)
- Verifica que el DataFrame no esté vacío
- Confirma ausencia de nulos y duplicados
- Por columna: sin infinitos (numéricas), sin strings vacíos (texto), rango de fechas (fechas)
- El pipeline **no guarda** si la validación falla

### 5. Guardado (`guardado.py`)
- Guarda el dataset limpio en `data/processed/data_clean.csv`
- Solo se ejecuta si la validación fue exitosa

---

## Cómo ejecutar

```bash
# Asegurarse de tener data.csv en la raíz del proyecto
python pipeline.py
```

Ajustar `FUENTE_PATH` e `ID_COL` en `pipeline.py` si el nombre del archivo o la columna ID cambian.

---

## Columnas finales del dataset procesado

```
UserID, Age, Gender, VRHeadset, Duration, MotionSickness,
ImmersionLevel, AgeGroup, DurationCategory, ComfortScore
```

**Shape final:** 1 000 filas × 10 columnas

---

## Decisiones de diseño

- **Detección automática de columnas:** ningún módulo tiene nombres de columnas hardcodeados; la limpieza y validación se adaptan a cualquier CSV.
- **Separación de responsabilidades:** cada archivo tiene una única función dentro del pipeline, lo que facilita el mantenimiento y la reutilización.
- **Validación como control de calidad:** el dataset procesado solo se persiste si pasa todos los controles, garantizando integridad en etapas posteriores como carga a base de datos o análisis.
- **Logging unificado:** todas las etapas escriben al mismo archivo `logs/pipeline.log`, permitiendo trazabilidad completa de cada ejecución.
