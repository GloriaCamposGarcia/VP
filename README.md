# SISTEMA DE CUMPLIMIENTO PLD/AML - ANÁLISIS NO SUPERVISADO

El sistema procesa información proveniente de herramientas de recolección OSINT y transacciones, consolidando perfiles semánticos de entidades, descubriendo vínculos ocultos de similitud semántica mediante embeddings locales, identificando anomalías e inusualidades en su información de forma no supervisada, y modelando la topología de red relacional con capas convolucionales de grafos (GNN) y comunidades estructurales.

---

## 1. ESTRUCTURA DE CARPETAS

La distribución de archivos del proyecto se organiza de la siguiente manera:
- **`data/raw/`**: Contiene los archivos CSV de entrada crudos (`evidence_items.csv` y `entity_source_results.csv`).
- **`data/processed/`**: Almacena los resultados intermedios y finales del procesamiento, embeddings en caché, métricas y los modelos serializados (`.pkl`, y `.parquet`).
- **`src/`**: Carpeta contenedora de los módulos lógicos del sistema:
  - `config.py`: Carga la configuración del archivo `.env` e inicializa el logging.
  - `data_processing.py`: Realiza la limpieza, normalización y extracción relacional inicial de entidades.
  - `embeddings_clustering.py`: Módulo 1. Genera embeddings de perfiles de entidad, calcula similitudes coseno para encontrar vínculos semánticos ocultos y computa anomalías latentes.
  - `anomaly_detection.py`: Módulo 2. Detecta anomalías cuantitativas en variables operacionales usando Isolation Forest, One-Class SVM y Local Outlier Factor de forma no supervisada.
  - `graph_analysis.py`: Módulo 3 (Plus de Valor). Analiza la red relacional e inyecta vínculos semánticos, calcula PageRank y detecta comunidades Louvain.
  - `pyspark_scaler.py`: Proporciona un pipeline escalable en PySpark para procesamientos transaccionales masivos.
- **`scripts/`**: Contiene scripts ejecutables del proyecto:
  - `run_benchmarking.py`: Orquesta la consolidación de métricas de agrupamiento, anomalías y estadísticas del grafo.
- **`docs/`**: Contiene reportes técnicos e informativos sobre la justificación regulatoria.
- **`tests/`**: Pruebas unitarias para validar las funciones esenciales del sistema.
- **`.env`**: Archivo de configuración local para variables de entorno.
- **`pld_aml_system.log`**: Registro continuo de actividades y errores del sistema.

---

## 2. REQUISITOS E INSTALACIÓN

### Requisitos Previos
El sistema requiere **Python 3.10 o superior** (probado con Python 3.14). Para el uso del módulo de escalabilidad PySpark, se requiere la presencia local de Java (JDK).

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

Los módulos deben ejecutarse desde la raíz del proyecto para asegurar la correcta resolución del path del módulo (`src`):

### Paso 1: Procesamiento de Datos
Se consolidan las entidades únicas, las variables operativas y las relaciones implícitas:
```bash
python -m src.data_processing
```

### Paso 2: Generación de Embeddings, Similitudes y Anomalías (Módulo 1)
Genera la representación vectorial de cada perfil de entidad, calcula vínculos de alta similitud de coseno, los inyecta en el grafo y calcula anomalías no supervisadas de embeddings:
```bash
python -m src.embeddings_clustering
```

### Paso 3: Análisis de Relaciones Ocultas y Grafos
Calcula PageRank, detecta comunidades Louvain y aplica la capa convolucional GNN sobre el grafo relacional y semántico consolidado:
```bash
python -m src.graph_analysis
```

### Paso 4: Detección No Supervisada de Anomalías Operacionales
Evalúa algoritmos no supervisados de detección de outliers (Isolation Forest, OneClassSVM, LOF) sobre las características cuantitativas de la población:
```bash
python -m src.anomaly_detection
```

### Paso 5: Visualización de Grafos Individuales (Opcional)
Genera en paralelo las visualizaciones ego concéntricas de red para cada entidad con conexiones en `data/processed/entity_graphs/`:
```bash
python -m src.visualization
```

### Paso 6: Escalabilidad PySpark
Ejecuta la consolidación paralela masiva de registros:
```bash
python -m src.pyspark_scaler
```

---

## 4. ORQUESTADOR DE REPORTING DE PATRONES

Para generar una vista consolidada en formato Markdown con las métricas de clustering, conteos de anomalías inusuales y la topología estructural del grafo relacional, ejecutar:

```bash
python scripts/run_benchmarking.py
```

El resultado final se imprime en consola y se guarda como documento formal en [benchmarking_report.md](file:///c:/Users/gloca/OneDrive/Desktop/Proyectos/Repositorios/Vincluos-Patrones/data/processed/benchmarking_report.md).

---

## 5. EJECUCIÓN DE PRUEBAS UNITARIAS

Para validar la estabilidad del sistema mediante el framework de pruebas de Python:

```bash
python -m unittest discover -s tests
```
