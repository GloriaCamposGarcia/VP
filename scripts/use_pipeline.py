import os
import sys
from pathlib import Path

# Se añade la raíz del proyecto al path de búsqueda de módulos
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Se establece el modo de uso/inferencia en las variables de entorno antes de importar módulos
os.environ["PIPELINE_MODE"] = "use"

from src.config import logger
from src.data_processing import prepare_pipeline
from src.pyspark_scaler import execute_pyspark_pipeline
from src.embeddings_clustering import execute_clustering_pipeline
from src.anomaly_detection import execute_anomaly_pipeline
from src.graph_analysis import execute_unsupervised_graph_analysis
from scripts.run_benchmarking import run_benchmarking_reporting

def run_use_pipeline():
    """
    Se ejecuta el pipeline completo de inferencia (uso), abarcando el procesamiento, escalado,
    agrupamiento semántico, detección de anomalías y generación de reportes y visualizaciones.
    """
    logger.info("==================================================")
    logger.info("INICIO DE PIPELINE DE USO / INFERENCIA PLD/AML (MLOps)")
    logger.info("==================================================")
    
    # 1. Se ejecuta el procesamiento de datos crudos para la nueva corrida
    logger.info("Paso 1: Procesamiento de datos y catalogación de la corrida en progreso.")
    prepare_pipeline()
    
    # 2. Se realiza el escalado de características con PySpark
    logger.info("Paso 2: Escalado PySpark en progreso.")
    execute_pyspark_pipeline()
    
    # 3. Se ejecuta la inferencia de clustering semántico y embeddings
    logger.info("Paso 3: Inferencia de clustering semántico en progreso.")
    execute_clustering_pipeline()
    
    # 4. Se ejecuta la inferencia de los modelos de detección de anomalías
    logger.info("Paso 4: Inferencia de modelos de anomalías en progreso.")
    execute_anomaly_pipeline()
    
    # 5. Se realiza el análisis de grafos y propagación GNN
    logger.info("Paso 5: Análisis de grafos y propagación GNN en progreso.")
    execute_unsupervised_graph_analysis()
    
    # 6. Se genera el reporte final de inferencia y visualizaciones
    logger.info("Paso 6: Consolidación de reporte y visualizaciones en progreso.")
    run_benchmarking_reporting()
    
    logger.info("==================================================")
    logger.info("PIPELINE DE USO / INFERENCIA FINALIZADO CON ÉXITO.")
    logger.info("==================================================")

if __name__ == '__main__':
    run_use_pipeline()
