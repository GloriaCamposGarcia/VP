import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from src.config import logger, DATA_PROCESSED_DIR

def load_data_for_anomalies() -> pd.DataFrame:
    """
    Carga el dataset de entidades consolidadas para el análisis de anomalías.
    """
    data_path = DATA_PROCESSED_DIR / 'consolidated_entities.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entidades en {data_path}.")
    
    df = pd.read_csv(data_path)
    logger.info(f"Dataset cargado con {len(df)} registros para detección de anomalías.")
    return df

def run_anomaly_benchmarking(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    Implementa Isolation Forest, One-Class SVM y Local Outlier Factor (LOF) de manera
    puramente no supervisada sobre las variables operativas de comportamiento de las entidades.
    """
    features = ['evidence_items', 'max_identity_score', 'sources_with_hallazgo']
    X = df[features].copy()
    
    logger.info(f"Variables cuantitativas para el modelado de anomalías: {features}")
    
    # Normalizar variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Proporción de contaminación por defecto (5% de la población)
    contamination_ratio = 0.05
    logger.info(f"Ratio de contaminación fijado para detección: {contamination_ratio:.4f}")
    
    # Inicializar modelos no supervisados
    models = {
        'IsolationForest': IsolationForest(
            contamination=contamination_ratio, 
            random_state=42, 
            n_jobs=-1
        ),
        'OneClassSVM': OneClassSVM(
            nu=contamination_ratio, 
            kernel='rbf', 
            gamma='scale'
        ),
        'LocalOutlierFactor': LocalOutlierFactor(
            contamination=contamination_ratio, 
            n_jobs=-1
        )
    }
    
    results = []
    trained_models = {}
    df_enriched = df.copy()
    
    # Entrenamiento y predicción en todo el conjunto de datos
    for name, clf in models.items():
        logger.info(f"Ejecutando modelo no supervisado: {name}...")
        t0 = time.time()
        
        if name == 'LocalOutlierFactor':
            # Local Outlier Factor en modo outlier predice directamente con fit_predict
            raw_pred = clf.fit_predict(X_scaled)
            scores = -clf.negative_outlier_factor_
        else:
            clf.fit(X_scaled)
            raw_pred = clf.predict(X_scaled)
            scores = -clf.score_samples(X_scaled)
            
        elapsed_time = time.time() - t0
        
        # Convertir a binario (1 para anomalía, 0 para inlier)
        y_pred = np.where(raw_pred == -1, 1, 0)
        num_anomalies = int(np.sum(y_pred))
        
        # Guardar predicciones y scores en el DataFrame
        df_enriched[f'anomaly_{name.lower()}'] = y_pred
        df_enriched[f'anomaly_score_{name.lower()}'] = scores
        
        results.append({
            'Algoritmo': name,
            'Num_Anomalias_Detectadas': num_anomalies,
            'Tiempo_Ejecucion_Segs': elapsed_time
        })
        
        trained_models[name] = clf
        logger.info(f"Modelo {name} completado. Anomalías detectadas: {num_anomalies}, Tiempo: {elapsed_time:.2f}s")
        
    df_bench = pd.DataFrame(results)
    return df_bench, trained_models, df_enriched

def execute_anomaly_pipeline() -> pd.DataFrame:
    """
    Ejecuta el pipeline de detección de anomalías y exporta las métricas e indicadores.
    """
    df = load_data_for_anomalies()
    df_bench, trained_models, df_enriched = run_anomaly_benchmarking(df)
    
    # Exportar los datos de las entidades con sus respectivas anomalías mapeadas
    df_enriched.to_csv(DATA_PROCESSED_DIR / 'consolidated_entities.csv', index=False)
    logger.info(f"Catálogo de entidades actualizado con anomalías en {DATA_PROCESSED_DIR / 'consolidated_entities.csv'}")
    
    # Exportar resultados de benchmarking
    df_bench.to_csv(DATA_PROCESSED_DIR / 'anomaly_metrics.csv', index=False)
    
    # Guardar el mejor modelo (IsolationForest)
    best_model = trained_models['IsolationForest']
    joblib.dump(best_model, DATA_PROCESSED_DIR / 'best_anomaly_model.pkl')
    logger.info("Modelo de anomalías IsolationForest serializado y guardado.")
    
    logger.info("\n--- METRICAS DE DETECCION DE ANOMALIAS (UNSUPERVISED) ---")
    logger.info(f"\n{df_bench.to_string(index=False)}")
    
    return df_bench

if __name__ == '__main__':
    execute_anomaly_pipeline()
