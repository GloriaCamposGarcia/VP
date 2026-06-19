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
    Se realiza la carga del conjunto de datos de entidades consolidadas para el análisis de anomalías.
    """
    data_path = RUN_DIR / 'consolidated_entities.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entidades en {data_path}.")
    
    df = pd.read_csv(data_path)
    logger.info(f"Datos cargados para anomalías. Total registros: {len(df)}")
    return df

def execute_anomaly_pipeline() -> pd.DataFrame:
    """
    Se ejecuta el flujo completo de detección de anomalías y se exportan las métricas obtenidas.
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
        logger.info("Modo de entrenamiento activo. Ajuste de StandardScaler y modelos de anomalías en progreso.")
        
        # 1. Se ajusta el modelo StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)
        logger.info("StandardScaler guardado en: shared/models/scaler.pkl")
        
        # 2. Se realiza el entrenamiento de los modelos no supervisados con ratio de contaminación
        contamination_ratio = 0.05
        logger.info(f"Parámetro de contaminación: {contamination_ratio:.4f}")
        
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
                novelty=True,
                n_jobs=-1
            )
        }
        
        trained_models = {}
        for name, clf in models.items():
            logger.info(f"Entrenamiento de modelo {name} en progreso.")
            t0 = time.time()
            clf.fit(X_scaled)
            elapsed_time = time.time() - t0
            
            # Se calculan las predicciones y scores de anomalía
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
            
        # Se realiza la persistencia de los modelos entrenados
        joblib.dump(trained_models['IsolationForest'], if_path)
        joblib.dump(trained_models['OneClassSVM'], svm_path)
        joblib.dump(trained_models['LocalOutlierFactor'], lof_path)
        
        # Se guarda una copia del Isolation Forest en el directorio local de auditoría
        joblib.dump(trained_models['IsolationForest'], RUN_DIR / 'best_anomaly_model.pkl')
        logger.info("Modelos de anomalías guardados en: shared/models/")

    else: # PIPELINE_MODE == "use"
        logger.info("Modo de uso/inferencia activo. Carga de StandardScaler y modelos en progreso.")
        
        if not scaler_path.exists() or not if_path.exists() or not svm_path.exists() or not lof_path.exists():
            raise FileNotFoundError(
                "Faltan modelos en el registro compartido. Debe ejecutar el entrenamiento: python scripts/train_pipeline.py"
            )
            
        scaler = joblib.load(scaler_path)
        models = {
            'IsolationForest': joblib.load(if_path),
            'OneClassSVM': joblib.load(svm_path),
            'LocalOutlierFactor': joblib.load(lof_path)
        }
        
        # Se ejecuta la inferencia aplicando la transformación del scaler cargado sin fit
        X_scaled = scaler.transform(X)
        
        for name, clf in models.items():
            logger.info(f"Inferencia con modelo {name} en progreso.")
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
            
        # Se persiste localmente el Isolation Forest cargado
        joblib.dump(models['IsolationForest'], RUN_DIR / 'best_anomaly_model.pkl')

    df_bench = pd.DataFrame(results)
    
    # Se exporta el catálogo de entidades enriquecido con indicadores
    df_enriched.to_csv(RUN_DIR / 'consolidated_entities.csv', index=False)
    logger.info(f"Catálogo con anomalías guardado en: {RUN_DIR / 'consolidated_entities.csv'}")
    
    # Se persisten los resultados del benchmarking cuantitativo
    df_bench.to_csv(RUN_DIR / 'anomaly_metrics.csv', index=False)
    
    logger.info("\n--- MÉTRICAS DE DETECCIÓN DE ANOMALÍAS ---")
    logger.info(f"\n{df_bench.to_string(index=False)}")
    
    return df_bench

if __name__ == '__main__':
    execute_anomaly_pipeline()
