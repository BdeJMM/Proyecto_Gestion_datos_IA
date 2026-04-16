# Predicción y Análisis de Gastos Personales

Este proyecto tiene como objetivo analizar y predecir el comportamiento de los gastos personales mediante un modelo de IA, utilizando variables como historial de gastos, categorías de consumo y presupuesto mensual, con el fin de apoyar la toma de decisiones financieras.

---

## Componentes del sistema

- **Scripts de procesamiento**: ingesta, limpieza, transformación y validación de datos.  
- **Base de datos PostgreSQL**: para la carga y consulta estructurada de los datasets.  
- **Modelo de IA (scikit-learn)**: predicción de gasto y detección de exceso de presupuesto.  
- **Metabase (opcional)**: dashboard de visualización de resultados.  
- **Documentación**: diseño técnico completo + planificación del proyecto.  

---

## Tecnologías utilizadas

- Python 3  
- Pandas / Scikit-learn  
- PostgreSQL  
- Docker  
- Git / GitHub  
- Trello / Jira (planificación)  

---

## Pipeline implementado

| Etapa | Descripción |
|-------|-------------|
| 1. Diseño e instalación | Definición de estructura del proyecto, entorno y herramientas |
| 2. Ingesta | Lectura desde CSV o generación de datos sintéticos |
| 3. Limpieza | Eliminación de duplicados, tratamiento de nulos y validación de tipos |
| 4. Transformación | Creación de variables como gasto mensual, gasto por categoría y desviación del presupuesto |
| 5. Validación | Revisión de coherencia, rangos y calidad de datos |
| 6. Carga en PostgreSQL | Almacenamiento del dataset limpio en base de datos |
| 7. Entrenamiento IA | Modelo predictivo para estimar gasto futuro o detectar exceso de presupuesto |
| 8. Evaluación | Métricas como MAE, RMSE o accuracy según modelo |
| 9. Visualización | Panel con análisis de gastos, tendencias y alertas (opcional) |

---

## 📂 Estructura del repositorio


gastos-personales-ia/
├── README.md
├── docs/
│ └── diseño_tecnico.pdf
├── scripts/
│ ├── ingesta.py
│ ├── limpieza.py
│ ├── transformacion.py
│ └── entrenamiento.py
├── data/
│ └── gastos.csv
├── dashboards/
│ └── dashboard_metabase.png
├── docker-compose.yml


---

## Cómo ejecutar el sistema (entorno ya instalado)

1. Clonar el repositorio  
   `git clone https://github.com/BdeJMM/Proyecto_Gestion_datos_IA[https://github.com/BdeJMM/Proyecto_Gestion_datos_IA]`

2. Entrar a la carpeta del proyecto  
   `cd gastos-personales-ia`

3. Ejecutar el pipeline manualmente por etapas  
   Ejemplo:  
   `python scripts/ingesta.py`  
   `python scripts/limpieza.py`  
   `python scripts/entrenamiento.py`

4. Visualizar los resultados y métricas desde consola o dashboard  

---

## Equipo

- Integrante 1 – Procesamiento, limpieza y entrenamiento 
- Integrante 2 – Modelado, visualización y documentación  
