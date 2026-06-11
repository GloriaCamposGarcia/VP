import sys
import os
import pandas as pd
from pathlib import Path
from typing import Optional

# Configurar variables de entorno de Spark para Windows
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from src.config import logger, DATA_RAW_DIR, DATA_PROCESSED_DIR

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, IntegerType, StringType
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False

class PySparkAMLScaler:
    """
    Proporciona un pipeline escalable utilizando PySpark para procesamiento transaccional masivo
    de entidades y cálculo de agregaciones OSINT de acuerdo a requerimientos Fintech.
    """
    def __init__(self):
        self.spark: Optional[SparkSession] = None

    def initialize_spark(self) -> bool:
        """
        Inicializa la sesión local de Spark si PySpark está instalado y configurado.
        Retorna True si la inicialización es exitosa, False en caso contrario.
        """
        if not HAS_PYSPARK:
            logger.warning("La biblioteca 'pyspark' no está disponible en este entorno.")
            return False
        try:
            logger.info("Inicializando sesión local de PySpark...")
            self.spark = SparkSession.builder \
                .appName("AML_Scalable_Aggregation") \
                .config("spark.sql.shuffle.partitions", "4") \
                .master("local[*]") \
                .getOrCreate()
            logger.info(f"Sesión de Spark inicializada con versión: {self.spark.version}")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar Spark: {e}")
            return False

    def scale_entity_aggregation(self) -> bool:
        """
        Demuestra la agregación escalable en paralelo cargando los archivos CSV crudos
        y consolidándolos con funciones nativas de PySpark SQL.
        """
        if not self.spark:
            logger.warning("Spark no está inicializado. No se puede ejecutar el procesamiento escalable.")
            return False
            
        logger.info("Ejecutando agregación escalable en PySpark...")
        try:
            # 1. Cargar datasets usando Pandas para evitar errores de alineación por comas en JSON
            logger.info("Cargando datasets crudos con Pandas...")
            df_sources_pd = pd.read_csv(DATA_RAW_DIR / 'entity_source_results.csv')
            df_evidence_pd = pd.read_csv(DATA_RAW_DIR / 'evidence_items.csv')
            
            # Normalizar tipos de datos para Spark
            for df_tmp in [df_sources_pd, df_evidence_pd]:
                for col in df_tmp.select_dtypes(include=['object']).columns:
                    df_tmp[col] = df_tmp[col].fillna("").astype(str)
                    
            # Crear DataFrames de Spark
            logger.info("Convirtiendo DataFrames a PySpark...")
            df_sources = self.spark.createDataFrame(df_sources_pd)
            df_evidence = self.spark.createDataFrame(df_evidence_pd)
            
            logger.info("Esquema de resultados de fuentes cargado en Spark:")
            df_sources.printSchema()
            
            # 2. Agrupación y cálculo de variables en paralelo
            # Fuentes evaluadas y fuentes con hallazgos
            # Se convierte evidence_count a numérico de manera segura
            df_sources_agg = df_sources.withColumn("evidence_count_num", F.col("evidence_count").cast(IntegerType())).groupBy("entity_id").agg(
                F.count("source_id").alias("sources_evaluated"),
                F.sum(F.when(F.col("evidence_count_num") > 0, 1).otherwise(0)).alias("sources_with_hallazgo")
            )
            
            # Evidencias máximas e ítems de revisión
            df_evidence_agg = df_evidence.withColumn("identity_score_num", F.col("identity_score").cast(DoubleType())).groupBy("entity_id").agg(
                F.max("identity_score_num").alias("max_identity_score"),
                F.count("evidence_id").alias("evidence_items"),
                F.sum(F.when(F.col("review_required").cast("string").rlike("(?i)true|1"), 1).otherwise(0)).alias("review_items")
            )
            
            # 3. Join en paralelo para consolidación
            df_consolidated = df_sources_agg.join(df_evidence_agg, on="entity_id", how="full")
            
            # Rellenar nulos de agregación
            df_consolidated = df_consolidated.na.fill({
                "sources_evaluated": 0,
                "sources_with_hallazgo": 0,
                "max_identity_score": 0.0,
                "evidence_items": 0,
                "review_items": 0
            })
            
            # Asignar lógica de decisión general
            df_consolidated = df_consolidated.withColumn(
                "overall_decision",
                F.when(F.col("review_items") > 0, "needs_review")
                .otherwise(F.when(F.col("evidence_items") > 0, "accepted").otherwise("no_match"))
            )
            
            # Mostrar una muestra en consola
            logger.info("Muestra de agregaciones procesadas en PySpark:")
            df_consolidated.show(5)
            
            # 4. Guardar los resultados en formato optimizado Parquet para almacenamiento masivo
            output_parquet_path = str(DATA_PROCESSED_DIR / 'pyspark_entities.parquet')
            logger.info(f"Guardando resultados escalables en Parquet: {output_parquet_path}")
            df_consolidated.write.mode("overwrite").parquet(output_parquet_path)
            
            return True
        except Exception as e:
            logger.error(f"Error durante la ejecución de agregación en Spark: {e}")
            return False
            
    def close_spark(self):
        """
        Cierra la sesión activa de Spark para liberar recursos del clúster.
        """
        if self.spark:
            self.spark.stop()
            logger.info("Sesión de PySpark cerrada.")

def execute_pyspark_pipeline() -> bool:
    """
    Orquesta la ejecución del pipeline escalable en PySpark.
    """
    scaler = PySparkAMLScaler()
    success = False
    if scaler.initialize_spark():
        success = scaler.scale_entity_aggregation()
        scaler.close_spark()
    return success

if __name__ == '__main__':
    execute_pyspark_pipeline()
