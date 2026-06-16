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

from src.config import logger, RUN_DIR, SHARED_DIR, PIPELINE_MODE

def load_data_for_anomalies() -> pd.DataFrame:
    """
    Carga el dataset de entidades consolidadas para el análisis de anomalías.
    """
    data_path = RUN_DIR / 'consolidated_entities.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entidades en {data_path}.")
    
    df = pd.read_csv(data_path)
    logger.info(f"Dataset cargado con {len(df)} registros para detección de anomalías.")
    return df

def execute_anomaly_pipeline() -> pd.DataFrame:
    """
    Ejecuta el pipeline de detección de anomalías y exporta las métricas e indicadores.
    """
    df = load_data_for_anomalies()
    
    features = ['evidence_items', 'max_identity_score', 'sources_with_hallazgo']
    X = df[features].copy()
    
    scaler_path = SHARED_DIR / "models" / "scaler.pkl"
    if_path = SHARED_DIR / "models" / "isolation_forest.pkl"
    svm_path = SHARED_DIR / "models" / "one_class_svm.pkl"
    lof_path = SHARED_DIR / "models" / "local_outlier_factor.pkl"

    df_enriched = df.copy()
    results = []

    if PIPELINE_MODE == "train":
        logger.info("Fase de ENTRENAMIENTO activa. Ajustando StandardScaler y modelos de anomalías...")
        
        # 1. Ajustar scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)
        logger.info("StandardScaler fiteado y guardado en shared/models/scaler.pkl")
        
        # 2. Entrenar modelos con novelty=True para LOF
        contamination_ratio = 0.05
        logger.info(f"Ratio de contaminación para entrenamiento: {contamination_ratio:.4f}")
        
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
                novelty=True, # Permitir predict inductivo
                n_jobs=-1
            )
        }
        
        trained_models = {}
        for name, clf in models.items():
            logger.info(f"Entrenando modelo no supervisado: {name}...")
            t0 = time.time()
            clf.fit(X_scaled)
            elapsed_time = time.time() - t0
            
            # Predicciones e indicadores
            raw_pred = clf.predict(X_scaled)
            scores = -clf.score_samples(X_scaled)
            y_pred = np.where(raw_pred == -1, 1, 0)
            num_anomalies = int(np.sum(y_pred))
            
            df_enriched[f'anomaly_{name.lower()}'] = y_pred
            df_enriched[f'anomaly_score_{name.lower()}'] = scores
            
            results.append({
                'Algoritmo': name,
                'Num_Anomalias_Detectadas': num_anomalies,
                'Tiempo_Ejecucion_Segs': elapsed_time
            })
            
            trained_models[name] = clf
            
        # Serializar modelos
        joblib.dump(trained_models['IsolationForest'], if_path)
        joblib.dump(trained_models['OneClassSVM'], svm_path)
        joblib.dump(trained_models['LocalOutlierFactor'], lof_path)
        
        # También guardar el mejor modelo en RUN_DIR por auditoría local
        joblib.dump(trained_models['IsolationForest'], RUN_DIR / 'best_anomaly_model.pkl')
        logger.info("Modelos de anomalías serializados y guardados en shared/models/")

    else: # PIPELINE_MODE == "use"
        logger.info("Fase de USO/INFERENCIA activa. Cargando StandardScaler y modelos de anomalías...")
        
        if not scaler_path.exists() or not if_path.exists() or not svm_path.exists() or not lof_path.exists():
            raise FileNotFoundError(
                "Faltan modelos en el registro compartido. Debe ejecutar el pipeline de entrenamiento primero: "
                "python scripts/train_pipeline.py"
            )
            
        scaler = joblib.load(scaler_path)
        models = {
            'IsolationForest': joblib.load(if_path),
            'OneClassSVM': joblib.load(svm_path),
            'LocalOutlierFactor': joblib.load(lof_path)
        }
        
        # Inferencia: aplicar transform sin fit
        X_scaled = scaler.transform(X)
        
        for name, clf in models.items():
            logger.info(f"Inferencia con modelo: {name}...")
            t0 = time.time()
            raw_pred = clf.predict(X_scaled)
            scores = -clf.score_samples(X_scaled)
            elapsed_time = time.time() - t0
            
            y_pred = np.where(raw_pred == -1, 1, 0)
            num_anomalies = int(np.sum(y_pred))
            
            df_enriched[f'anomaly_{name.lower()}'] = y_pred
            df_enriched[f'anomaly_score_{name.lower()}'] = scores
            
            results.append({
                'Algoritmo': name,
                'Num_Anomalias_Detectadas': num_anomalies,
                'Tiempo_Ejecucion_Segs': elapsed_time
            })
            
        # Guardar el IsolationForest cargado localmente en RUN_DIR por compatibilidad
        joblib.dump(models['IsolationForest'], RUN_DIR / 'best_anomaly_model.pkl')

    df_bench = pd.DataFrame(results)
    
    # Exportar los datos actualizados de las entidades
    df_enriched.to_csv(RUN_DIR / 'consolidated_entities.csv', index=False)
    logger.info(f"Catálogo de entidades actualizado con anomalías en {RUN_DIR / 'consolidated_entities.csv'}")
    
    # Exportar resultados de benchmarking
    df_bench.to_csv(RUN_DIR / 'anomaly_metrics.csv', index=False)
    
    logger.info("\n--- METRICAS DE DETECCION DE ANOMALIAS (UNSUPERVISED) ---")
    logger.info(f"\n{df_bench.to_string(index=False)}")
    
    return df_bench

if __name__ == '__main__':
    execute_anomaly_pipeline()
