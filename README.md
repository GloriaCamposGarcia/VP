# SISTEMA DE CUMPLIMIENTO AML-OSINT

El sistema procesa información proveniente de herramientas de recolección OSINT y transacciones, consolidando perfiles semánticos de entidades, descubriendo vínculos ocultos de similitud semántica mediante embeddings locales e identificando anomalías y patrones  de forma no supervisada.

Se cuenta con una fase de entrenamiento (ajuste de modelos de agrupamiento y detección de anomalías) y de una fase de inferencia o uso (aplicación de los modelos registrados sobre nuevos perfiles).

---

## 1. ESTRUCTURA DE CARPETAS

La distribución de archivos del proyecto se organiza de la siguiente manera:
- **`data/raw/`**: Contiene los archivos CSV de entrada crudos (`evidence_items.csv`, `entity_source_results.csv` y `entity_match_summary.csv`).
- **`data/processed/`**: Directorio de almacenamiento de datos procesados, el cual se subdivide en:
  - **`shared/`**: Contiene modelos y artefactos compartidos/registrados.
  - **`train/runs/`**: Contiene carpetas únicas de ejecución de entrenamiento denominadas con marca de tiempo (`run_YYYY-MM-DD_HH-MM-SS`).
  - **`use/runs/`**: Contiene carpetas únicas de ejecución de uso/inferencia denominadas con marca de tiempo (`run_YYYY-MM-DD_HH-MM-SS`).
- **`src/`**: Carpeta contenedora de los módulos lógicos del sistema:
  - `config.py`: Establece la ruta del proyecto, variables de entorno y el registro de eventos (logging).
  - `data_processing.py`: Realiza la limpieza, normalización y extracción relacional inicial de entidades.
  - `embeddings_clustering.py`: Módulo de agrupamiento semántico y representación vectorial (Sentence-Transformers y HDBSCAN/K-Means).
  - `anomaly_detection.py`: Módulo de detección de anomalías cuantitativas (Isolation Forest, One-Class SVM y Local Outlier Factor).
  - `graph_analysis.py`: Módulo de análisis topológico de grafos, PageRank y comunidades Louvain.
  - `chain_analysis.py`: Análisis de caminos relacionales de múltiples saltos (hasta 3 saltos) y bucles sospechosos (triangulaciones).
  - `visualization.py`: Genera de forma paralela las visualizaciones ego concéntricas de red para cada entidad con conexiones.
  - `pyspark_scaler.py`: Implementación escalable en PySpark para procesamientos masivos de entidades.
- **`scripts/`**: Ejecutables principales del pipeline:
  - `train_pipeline.py`: Ejecución del pipeline completo de entrenamiento y ajuste de modelos.
  - `use_pipeline.py`: Ejecución de la fase de uso aplicando modelos guardados.
  - `run_benchmarking.py`: Módulo de compilación del reporte de desempeño y topología.
- **`docs/`**: Contiene reportes técnicos adicionales y documentación de justificación.
- **`tests/`**: Pruebas unitarias para validar las funciones esenciales del sistema.

---

## 2. REQUISITOS E INSTALACIÓN

### Requisitos Previos
El sistema requiere **Python 3.10 o superior**. Para el uso del módulo de escalabilidad PySpark, se requiere la presencia local de Java (JDK).

### Instalación de Dependencias
Se recomienda configurar un entorno virtual para instalar los paquetes necesarios:

```bash
# Crear entorno virtual
python -m venv venv
# Activar entorno virtual (Windows)
.\venv\Scripts\activate

# Instalar dependencias requeridas
pip install pandas numpy scikit-learn sentence-transformers networkx torch torch-geometric pyspark python-dotenv
```

---

## 3. INSTRUCCIONES DE EJECUCIÓN

La ejecución de los pipelines se realiza desde la raíz del proyecto para asegurar la correcta resolución de rutas:

### Ejecución de la Fase de Entrenamiento
Se realiza el procesamiento, escalado con PySpark, ajuste de embeddings, entrenamiento de los modelos de detección de anomalías/clustering, análisis de grafos y generación del reporte de benchmarking de entrenamiento:
```bash
python scripts/train_pipeline.py
```
Los artefactos generados se guardan en un subdirectorio único bajo:
`data/processed/train/runs/run_<timestamp>/`

### Ejecución de la Fase de Uso / Inferencia
Se procesa el lote de datos de uso, se recuperan los modelos ajustados desde `data/processed/shared/models/` y se ejecuta la inferencia semántica y de detección de anomalías.
```bash
python scripts/use_pipeline.py
```
Los artefactos generados de uso se guardan en el subdirectorio:
`data/processed/use/runs/run_<timestamp>/`

---

## 4. REPORTES GENERADOS Y TRAZABILIDAD

El archivo `benchmarking_report.md` se genera automáticamente dentro del directorio de la corrida correspondiente tanto en la fase de entrenamiento como de uso. Este archivo contiene las estadísticas del agrupamiento, el análisis de anomalías, las comunidades de red (Louvain), y las rutas de caminos y bucles identificados en el análisis relacional.

---

## 5. EJECUCIÓN DE PRUEBAS UNITARIAS

Para validar la estabilidad del sistema mediante el framework de pruebas de Python:

```bash
python -m unittest discover -s tests
```
