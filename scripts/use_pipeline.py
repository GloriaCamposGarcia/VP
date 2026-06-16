import os
import sys
from pathlib import Path

# Añadir la raíz del proyecto al path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Establecer modo uso ANTES de importar config/módulos
os.environ["PIPELINE_MODE"] = "use"

from src.config import logger
from src.data_processing import prepare_pipeline
from src.pyspark_scaler import execute_pyspark_pipeline
from src.embeddings_clustering import execute_clustering_pipeline
from src.anomaly_detection import execute_anomaly_pipeline
from src.graph_analysis import execute_unsupervised_graph_analysis
from scripts.run_benchmarking import run_benchmarking_reporting

def run_use_pipeline():
    logger.info("==================================================")
    logger.info("INICIANDO PIPELINE DE USO / INFERENCIA PLD/AML")
    logger.info("==================================================")
    
    # 1. Procesamiento de datos crudos
    logger.info("Paso 1: Procesamiento de datos de la nueva corrida...")
    prepare_pipeline()
    
    # 2. PySpark Scaler
    logger.info("Paso 2: Escalado PySpark...")
    execute_pyspark_pipeline()
    
    # 3. Clustering y Embeddings (inferencia)
    logger.info("Paso 3: Inferencia de Clustering y Embeddings...")
    execute_clustering_pipeline()
    
    # 4. Detección de anomalías (inferencia)
    logger.info("Paso 4: Inferencia de Detección de Anomalías...")
    execute_anomaly_pipeline()
    
    # 5. Análisis topológico de grafos
    logger.info("Paso 5: Análisis de Grafos No Supervisado...")
    execute_unsupervised_graph_analysis()
    
    # 6. Reporte final e imágenes
    logger.info("Paso 6: Generación de imágenes y Reporte de Inferencia...")
    run_benchmarking_reporting()
    
    logger.info("==================================================")
    logger.info("PIPELINE DE USO / INFERENCIA COMPLETADO EXITOSAMENTE")
    logger.info("==================================================")

if __name__ == '__main__':
    run_use_pipeline()
