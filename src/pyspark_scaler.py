import sys
import os
import pandas as pd
from pathlib import Path
from typing import Optional

# Se configuran las variables de entorno de Spark para Windows
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from src.config import logger, DATA_RAW_DIR, DATA_PROCESSED_DIR, RUN_DIR

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, IntegerType, StringType
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False

class PySparkAMLScaler:
    """
    Se proporciona un pipeline escalable utilizando PySpark para el procesamiento transaccional masivo
    de entidades y el cálculo de agregaciones OSINT.
    """
    def __init__(self):
        self.spark: Optional[SparkSession] = None

    def initialize_spark(self) -> bool:
        """
        Se inicializa la sesión local de Spark si PySpark se encuentra instalado y configurado en el sistema.
        """
        if not HAS_PYSPARK:
            logger.warning("PySpark no disponible en el entorno actual.")
            return False
        try:
            logger.info("Inicialización de sesión local PySpark en progreso.")
            self.spark = SparkSession.builder \
                .appName("AML_Scalable_Aggregation") \
                .config("spark.sql.shuffle.partitions", "4") \
                .master("local[*]") \
                .getOrCreate()
            logger.info(f"Sesión de Spark activa (versión: {self.spark.version}).")
            return True
        except Exception as e:
            logger.error(f"Error en inicialización de Spark: {e}")
            return False

    def scale_entity_aggregation(self) -> bool:
        """
        Se realiza la agregación escalable en paralelo mediante la carga de archivos de origen CSV
        y su consolidación a través de funciones nativas de PySpark SQL.
        """
        if not self.spark:
            logger.warning("Spark no inicializado.")
            return False
            
        logger.info("Agregación escalable en Spark en progreso.")
        try:
            # 1. Cargar conjuntos de datos usando Pandas para prevenir errores de delimitación en JSON
            logger.info("Carga de conjuntos de datos crudos mediante Pandas.")
            df_sources_pd = pd.read_csv(DATA_RAW_DIR / 'entity_source_results.csv')
            df_evidence_pd = pd.read_csv(DATA_RAW_DIR / 'evidence_items.csv')
            df_match_pd = pd.read_csv(DATA_RAW_DIR / 'entity_match_summary.csv')
            
            # Normalización de los tipos de datos para compatibilidad con Spark
            for df_tmp in [df_sources_pd, df_evidence_pd, df_match_pd]:
                for col in df_tmp.select_dtypes(include=['object']).columns:
                    df_tmp[col] = df_tmp[col].fillna("").astype(str)
                    
            # DataFrames nativos de Spark
            logger.info("Conversión de DataFrames a Spark en progreso.")
            df_sources = self.spark.createDataFrame(df_sources_pd)
            df_evidence = self.spark.createDataFrame(df_evidence_pd)
            df_match = self.spark.createDataFrame(df_match_pd)
            
            logger.info("Esquema de resultados de fuentes:")
            df_sources.printSchema()
            
            # 2. Agrupación y cálculo de variables en paralelo
            df_sources_agg = df_sources.withColumn("evidence_count_num", F.col("evidence_count").cast(IntegerType())).groupBy("entity_id").agg(
                F.count("source_id").alias("sources_evaluated"),
                F.sum(F.when(F.col("evidence_count_num") > 0, 1).otherwise(0)).alias("sources_with_hallazgo")
            )
            
            df_evidence_agg = df_evidence.withColumn("identity_score_num", F.col("identity_score").cast(DoubleType())).groupBy("entity_id").agg(
                F.max("identity_score_num").alias("max_identity_score"),
                F.count("evidence_id").alias("evidence_items"),
                F.sum(F.when(F.col("review_required").cast("string").rlike("(?i)true|1"), 1).otherwise(0)).alias("review_items")
            )
            
            # 3. Se realiza el join en paralelo para la consolidación de atributos
            df_consolidated = df_sources_agg.join(df_evidence_agg, on="entity_id", how="full")
            
            # Se seleccionan y procesan los atributos del resumen de coincidencias
            df_match_sel = df_match.select(
                F.col("entity_id"),
                F.col("match_count").cast(IntegerType()).alias("match_count"),
                F.col("sources_hit")
            )
            df_consolidated = df_consolidated.join(df_match_sel, on="entity_id", how="full")
            
            # Se imputan valores nulos derivados del join
            df_consolidated = df_consolidated.na.fill({
                "sources_evaluated": 0,
                "sources_with_hallazgo": 0,
                "max_identity_score": 0.0,
                "evidence_items": 0,
                "review_items": 0,
                "match_count": 0,
                "sources_hit": ""
            })
            
            # Lógica para la decisión general
            df_consolidated = df_consolidated.withColumn(
                "overall_decision",
                F.when(F.col("review_items") > 0, "needs_review")
                .otherwise(F.when(F.col("evidence_items") > 0, "accepted").otherwise("no_match"))
            )
            
            # Resultados
            logger.info("Muestra de datos consolidados:")
            df_consolidated.show(5)
            
            # 4. Resultados en formato Parquet
            output_parquet_path = str(RUN_DIR / 'pyspark_entities.parquet')
            logger.info(f"Persistencia en Parquet: {output_parquet_path}")
            df_consolidated.write.mode("overwrite").parquet(output_parquet_path)
            
            return True
        except Exception as e:
            logger.error(f"Error en agregación Spark: {e}")
            return False
            
    def close_spark(self):
        """
        Se realiza el cierre de la sesión activa de Spark para la liberación de los recursos del sistema.
        """
        if self.spark:
            self.spark.stop()
            logger.info("Sesión de PySpark cerrada.")

def execute_pyspark_pipeline() -> bool:
    """
    Ejecución completa del pipeline escalable en PySpark.
    """
    scaler = PySparkAMLScaler()
    success = False
    if scaler.initialize_spark():
        success = scaler.scale_entity_aggregation()
        scaler.close_spark()
    return success

if __name__ == '__main__':
    execute_pyspark_pipeline()
