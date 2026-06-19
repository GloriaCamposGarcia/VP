import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, List
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

import joblib
from src.config import logger, RUN_DIR, HF_EMBEDDING_MODEL, EMBEDDING_BACKEND, SHARED_DIR, PIPELINE_MODE

def generate_embeddings(texts: pd.Series) -> Tuple[np.ndarray, str]:
    """
    Se realiza la generación de embeddings vectoriales para la colección de perfiles de texto.
    Se implementa un mecanismo de caché en disco para evitar re-cálculos redundantes.
    """
    cache_path = RUN_DIR / 'entity_embeddings_cache.npy'
    
    if cache_path.exists():
        logger.info(f"Carga de embeddings desde caché: {cache_path}")
        X = np.load(cache_path)
        return X, EMBEDDING_BACKEND

    logger.info(f"Generación de embeddings en progreso (backend: {EMBEDDING_BACKEND}).")
    texts_list = texts.fillna("").astype(str).tolist()

    if EMBEDDING_BACKEND in ['sentence-transformers', 'hf', 'sentencetransformers']:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Modelo SentenceTransformer: {HF_EMBEDDING_MODEL}")
            model = SentenceTransformer(HF_EMBEDDING_MODEL)
            # Se realiza la codificación vectorial
            embeddings = model.encode(texts_list, show_progress_bar=True, convert_to_numpy=True)
            # Se normaliza mediante norma L2 para equivalencia en distancia euclidiana
            X = normalize(embeddings, norm='l2', axis=1)
            np.save(cache_path, X)
            logger.info("Cálculo y guardado de embeddings completado.")
            return X, 'sentence-transformers'
        except Exception as e:
            logger.error(f"Error en SentenceTransformers: {e}. Fallback a TF-IDF.")

    # Fallback a TF-IDF
    logger.info("Cálculo de representación TF-IDF en progreso.")
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char_wb', min_df=2)
    X_sparse = vectorizer.fit_transform(texts_list)
    X = normalize(X_sparse.toarray(), norm='l2', axis=1)
    np.save(cache_path, X)
    logger.info("Cálculo y guardado de embeddings TF-IDF completado.")
    return X, 'tfidf'

def reduce_dimensions(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Se aplica reducción de dimensionalidad mediante PCA y t-SNE sobre la matriz de embeddings.
    """
    logger.info("PCA (2 componentes) en progreso.")
    t0 = time.time()
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    logger.info(f"PCA completado en {time.time() - t0:.2f}s.")

    logger.info("t-SNE (2 componentes) en progreso.")
    t0 = time.time()
    # Se ajusta la perplejidad dinámicamente según el tamaño de la muestra
    n_samples = X.shape[0]
    perp = min(30, max(5, n_samples - 1))
    tsne = TSNE(n_components=2, random_state=42, max_iter=300, perplexity=perp, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)
    logger.info(f"t-SNE completado en {time.time() - t0:.2f}s (perplejidad: {perp}).")

    return X_pca, X_tsne

def calculate_cohesion(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Se calcula la métrica de cohesión interna para los clústeres estructurados.
    Representa el promedio de las distancias euclidianas de cada punto a su centroide asignado.
    """
    unique_labels = np.unique(labels)
    # Se excluye la etiqueta de ruido (-1) si está presente
    unique_labels = [l for l in unique_labels if l != -1]
    
    if not unique_labels:
        return np.nan
        
    distances = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = cluster_points.mean(axis=0)
        # Se calcula la distancia euclidiana de cada punto al centroide
        dist = np.linalg.norm(cluster_points - centroid, axis=1)
        distances.extend(dist.tolist())
        
    return float(np.mean(distances))

def run_clustering_benchmarking(X: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Se ejecutan los algoritmos de K-Means y HDBSCAN para realizar un benchmarking cuantitativo.
    Se evalúan el coeficiente de silueta y la cohesión.
    """
    logger.info("Benchmarking de clustering en progreso.")
    results = {}
    cluster_labels = {}

    n_samples = X.shape[0]
    n_clusters_kmeans = min(5, n_samples) if n_samples > 1 else 1

    # 1. K-Means
    logger.info(f"Entrenamiento K-Means en progreso (clústeres: {n_clusters_kmeans}).")
    t0 = time.time()
    if n_samples > 1:
        kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
        km_labels = kmeans.fit_predict(X)
        km_time = time.time() - t0
        cluster_labels['K-Means'] = km_labels
        
        km_sil = float(silhouette_score(X, km_labels, metric='euclidean'))
        km_cohesion = calculate_cohesion(X, km_labels)
    else:
        km_labels = np.zeros(n_samples, dtype=int)
        km_time = time.time() - t0
        cluster_labels['K-Means'] = km_labels
        km_sil = np.nan
        km_cohesion = np.nan

    results['K-Means'] = {
        'Silhouette': km_sil,
        'Cohesion': km_cohesion,
        'Time_Secs': km_time,
        'Noise_Count': 0,
        'Num_Clusters': len(np.unique(km_labels))
    }
    logger.info(f"K-Means finalizado. Silueta: {km_sil:.4f}, Cohesión: {km_cohesion:.4f}")

    # 2. HDBSCAN
    logger.info("Entrenamiento HDBSCAN en progreso.")
    t0 = time.time()
    # Se definen min_cluster_size y min_samples dinámicamente
    min_cluster = min(10, max(2, n_samples // 3))
    min_samp = min(5, max(1, min_cluster // 2))
    
    logger.info(f"HDBSCAN params: min_cluster_size={min_cluster}, min_samples={min_samp}")
    hdb = HDBSCAN(min_cluster_size=min_cluster, min_samples=min_samp, metric='euclidean')
    hdb_labels = hdb.fit_predict(X)
    hdb_time = time.time() - t0
    cluster_labels['HDBSCAN'] = hdb_labels

    # Métricas de evaluación para HDBSCAN
    non_noise_mask = hdb_labels != -1
    num_clusters = len([c for c in np.unique(hdb_labels) if c != -1])
    noise_count = int(np.sum(hdb_labels == -1))
    
    if num_clusters >= 2 and np.sum(non_noise_mask) > num_clusters:
        hdb_sil = float(silhouette_score(X[non_noise_mask], hdb_labels[non_noise_mask], metric='euclidean'))
        hdb_cohesion = calculate_cohesion(X, hdb_labels)
    else:
        hdb_sil = np.nan
        hdb_cohesion = np.nan

    results['HDBSCAN'] = {
        'Silhouette': hdb_sil,
        'Cohesion': hdb_cohesion,
        'Time_Secs': hdb_time,
        'Noise_Count': noise_count,
        'Num_Clusters': num_clusters
    }
    logger.info(f"HDBSCAN finalizado. Silueta: {hdb_sil:.4f}, Cohesión: {hdb_cohesion:.4f}, Ruido: {noise_count}")

    return results, cluster_labels

def execute_clustering_pipeline() -> pd.DataFrame:
    """
    Se ejecuta el flujo completo de agrupamiento semántico para la extracción de perfiles y vínculos.
    """
    entities_path = RUN_DIR / 'consolidated_entities.csv'
    evidence_path = RUN_DIR / 'processed_evidence_items.csv'
    
    if not entities_path.exists():
        raise FileNotFoundError(f"No se encontró {entities_path}. Ejecuta primero data_processing.py")
    if not evidence_path.exists():
        raise FileNotFoundError(f"No se encontró {evidence_path}. Ejecuta primero data_processing.py")

    df_entities = pd.read_csv(entities_path)
    df_evidence = pd.read_csv(evidence_path)
    
    logger.info("Agrupamiento de evidencias para perfiles de texto en progreso.")
    df_evidence['snippet_clean'] = df_evidence['snippet'].fillna('').astype(str).str.strip()
    df_evidence['reason_clean'] = df_evidence['review_reason'].fillna('').astype(str).str.strip()
    
    # Se combinan los campos de fragmento y contexto
    df_evidence['item_text'] = df_evidence.apply(
        lambda r: f"Fragmento: {r['snippet_clean']} | Motivo: {r['reason_clean']}" 
        if r['reason_clean'] else f"Fragmento: {r['snippet_clean']}", 
        axis=1
    )
    
    # Se realiza la agregación por identificador de entidad
    grouped_evidence = df_evidence.groupby('entity_id')['item_text'].apply(
        lambda x: " ; ".join(x.unique())
    ).rename('all_evidence')
    
    # Se asocia la información consolidada con el catálogo de entidades
    if 'predominant_cluster' in df_entities.columns:
        df_entities = df_entities.drop(columns=['predominant_cluster'])
    df_entities = df_entities.merge(grouped_evidence, on='entity_id', how='left')
    df_entities['all_evidence'] = df_entities['all_evidence'].fillna("Sin evidencias recopiladas").replace("", "Sin evidencias recopiladas")
    
    # Se genera el perfil textual descriptivo
    df_entities['entity_profile_text'] = df_entities.apply(
        lambda r: f"Entidad: {r['entity_name']} | Tipo: {r['entity_type']} | País: {r['country_code']} | Evidencias: {r['all_evidence']}",
        axis=1
    )
    
    df_entities['entity_profile_text'] = df_entities['entity_profile_text'].fillna("Sin perfil disponible")

    # Se obtienen embeddings vectoriales del perfil
    X, backend = generate_embeddings(df_entities['entity_profile_text'])
    n_entities = len(df_entities)
    
    kmeans_model_path = SHARED_DIR / "models" / "kmeans.pkl"
    pca_model_path = SHARED_DIR / "models" / "pca.pkl"
    iso_forest_model_path = SHARED_DIR / "models" / "iso_forest_embeddings.pkl"

    if PIPELINE_MODE == "train":
        logger.info("Modo de entrenamiento activo. Ajuste de modelos de clustering en progreso.")
        metrics, labels = run_clustering_benchmarking(X)
        
        # Ajuste de modelos definidos para persistencia
        n_clusters_kmeans = min(5, n_entities) if n_entities > 1 else 1
        kmeans_model = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
        kmeans_model.fit(X)
        
        pca_model = PCA(n_components=2, random_state=42)
        X_pca = pca_model.fit_transform(X)
        
        from sklearn.ensemble import IsolationForest
        iso_forest_model = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
        iso_forest_model.fit(X)
        
        # Se serializan los modelos en la ruta compartida de MLOps
        joblib.dump(kmeans_model, kmeans_model_path)
        joblib.dump(pca_model, pca_model_path)
        joblib.dump(iso_forest_model, iso_forest_model_path)
        logger.info(f"Modelos de clustering serializados en: {SHARED_DIR / 'models'}")
        
        # Se reduce la dimensionalidad mediante t-SNE para fines de visualización
        perp = min(30, max(5, n_entities - 1))
        tsne = TSNE(n_components=2, random_state=42, max_iter=300, perplexity=perp, n_jobs=-1)
        X_tsne = tsne.fit_transform(X)
        
        kmeans_labels = labels['K-Means']
        hdbscan_labels = labels['HDBSCAN']
        
        # Se obtienen los scores y predicciones del modelo de aislamiento (Isolation Forest)
        raw_scores = iso_forest_model.score_samples(X)
        df_entities['anomaly_score_embedding'] = -raw_scores
        df_entities['is_embedding_outlier'] = np.where(iso_forest_model.predict(X) == -1, 1, 0)
        
        # Se determina la distancia de cada muestra a su respectivo centroide
        distances = kmeans_model.transform(X)
        dist_to_centroid = np.array([distances[idx, cluster] for idx, cluster in enumerate(kmeans_labels)]) if n_entities > 1 else np.zeros(n_entities)
        df_entities['distance_to_centroid'] = dist_to_centroid

    else: # PIPELINE_MODE == "use"
        logger.info("Modo de uso/inferencia activo. Carga de modelos en progreso.")
        if not kmeans_model_path.exists() or not pca_model_path.exists() or not iso_forest_model_path.exists():
            raise FileNotFoundError(
                "Faltan modelos en el registro compartido. Ejecute primero el entrenamiento: python scripts/train_pipeline.py"
            )
            
        kmeans_model = joblib.load(kmeans_model_path)
        pca_model = joblib.load(pca_model_path)
        iso_forest_model = joblib.load(iso_forest_model_path)
        
        # Se ejecuta la inferencia con el modelo K-Means
        kmeans_labels = kmeans_model.predict(X)
        
        # Se realiza la inferencia de anomalías
        raw_scores = iso_forest_model.score_samples(X)
        df_entities['anomaly_score_embedding'] = -raw_scores
        df_entities['is_embedding_outlier'] = np.where(iso_forest_model.predict(X) == -1, 1, 0)
        
        # Se determina la distancia al centroide
        distances = kmeans_model.transform(X)
        dist_to_centroid = np.array([distances[idx, cluster] for idx, cluster in enumerate(kmeans_labels)]) if n_entities > 1 else np.zeros(n_entities)
        df_entities['distance_to_centroid'] = dist_to_centroid
        
        # Se aplica HDBSCAN sobre el conjunto actual de inferencia
        min_cluster = min(10, max(2, n_entities // 3))
        min_samp = min(5, max(1, min_cluster // 2))
        hdb = HDBSCAN(min_cluster_size=min_cluster, min_samples=min_samp, metric='euclidean')
        hdbscan_labels = hdb.fit_predict(X)
        
        X_pca = pca_model.transform(X)
        
        perp = min(30, max(5, n_entities - 1))
        tsne = TSNE(n_components=2, random_state=42, max_iter=300, perplexity=perp, n_jobs=-1)
        X_tsne = tsne.fit_transform(X)
        
        # Se calculan las métricas de monitoreo sobre el lote actual
        km_sil = float(silhouette_score(X, kmeans_labels, metric='euclidean')) if len(np.unique(kmeans_labels)) > 1 else np.nan
        km_cohesion = calculate_cohesion(X, kmeans_labels)
        
        non_noise_mask = hdbscan_labels != -1
        num_clusters_hdb = len([c for c in np.unique(hdbscan_labels) if c != -1])
        noise_count = int(np.sum(hdbscan_labels == -1))
        
        if num_clusters_hdb >= 2 and np.sum(non_noise_mask) > num_clusters_hdb:
            hdb_sil = float(silhouette_score(X[non_noise_mask], hdbscan_labels[non_noise_mask], metric='euclidean'))
            hdb_cohesion = calculate_cohesion(X, hdbscan_labels)
        else:
            hdb_sil = np.nan
            hdb_cohesion = np.nan
            
        metrics = {
            'K-Means': {
                'Silhouette': km_sil,
                'Cohesion': km_cohesion,
                'Time_Secs': 0.0,
                'Noise_Count': 0,
                'Num_Clusters': len(np.unique(kmeans_labels))
            },
            'HDBSCAN': {
                'Silhouette': hdb_sil,
                'Cohesion': hdb_cohesion,
                'Time_Secs': 0.0,
                'Noise_Count': noise_count,
                'Num_Clusters': num_clusters_hdb
            }
        }

    # Se guardan las etiquetas asignadas en el DataFrame
    df_entities['kmeans_cluster'] = kmeans_labels
    df_entities['hdbscan_cluster'] = hdbscan_labels
    df_entities['predominant_cluster'] = kmeans_labels
    
    df_entities['pca_x'] = X_pca[:, 0]
    df_entities['pca_y'] = X_pca[:, 1]
    df_entities['tsne_x'] = X_tsne[:, 0]
    df_entities['tsne_y'] = X_tsne[:, 1]
    
    # Se asignan etiquetas temáticas basadas en K-Means
    cluster_themes = {
        0: "Sanciones Financieras e Internacionales",
        1: "Personas de Alto Riesgo / PEPs",
        2: "SAT Listas Negras / Discrepancias Fiscales",
        3: "Listas de Alerta Locales / Medios Adversos",
        4: "Registros de Control General / Homónimos"
    }
    df_entities['kmeans_cluster_theme'] = df_entities['kmeans_cluster'].map(cluster_themes).fillna("Otros Hallazgos")
    
    # 1. Se detectan vínculos semánticos mediante similitud de coseno
    logger.info("Detección de vínculos semánticos implícitos en progreso.")
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(X)
    
    links = []
    for i in range(n_entities):
        for j in range(i + 1, n_entities):
            sim = float(similarity_matrix[i, j])
            if sim >= 0.70:
                ent_i = df_entities.iloc[i]
                ent_j = df_entities.iloc[j]
                links.append({
                    'source': ent_i['entity_id'],
                    'source_name': ent_i['entity_name'],
                    'target': ent_j['entity_id'],
                    'target_name': ent_j['entity_name'],
                    'relation_type': 'semantic_similarity',
                    'weight': sim
                })
    
    df_links = pd.DataFrame(links)
    if not df_links.empty:
        links_output_path = RUN_DIR / 'hidden_entity_links.csv'
        df_links.to_csv(links_output_path, index=False)
        logger.info(f"Vínculos semánticos guardados en: {links_output_path} (total: {len(df_links)})")
        
        # Se inyectan las relaciones semánticas en el conjunto relacional
        edges_path = RUN_DIR / 'entity_edges.csv'
        if edges_path.exists():
            df_edges = pd.read_csv(edges_path)
        else:
            df_edges = pd.DataFrame(columns=['source', 'target', 'relation_type', 'weight'])
            
        df_links_for_merge = df_links[['source', 'target', 'relation_type', 'weight']]
        df_edges_combined = pd.concat([df_edges, df_links_for_merge], ignore_index=True)
        # Se consolidan aristas tomando el valor máximo de similitud
        df_edges_combined = df_edges_combined.groupby(['source', 'target', 'relation_type'], as_index=False)['weight'].max()
        df_edges_combined.to_csv(edges_path, index=False)
        logger.info(f"Inyección de {len(df_links)} aristas semánticas en: {edges_path}")
    else:
        logger.info("Sin vínculos detectados (similitud >= 0.70).")
        pd.DataFrame(columns=['source', 'source_name', 'target', 'target_name', 'relation_type', 'weight']).to_csv(
            RUN_DIR / 'hidden_entity_links.csv', index=False
        )

    # Se guarda el catálogo consolidado de entidades
    df_entities.to_csv(RUN_DIR / 'consolidated_entities.csv', index=False)
    logger.info(f"Catálogo actualizado en: {RUN_DIR / 'consolidated_entities.csv'}")

    # Se estructuran las métricas comparativas en un DataFrame
    rows = []
    for alg, data in metrics.items():
        rows.append({
            'Algoritmo': alg,
            'Num_Clusters': data['Num_Clusters'],
            'Ruido_Detectado': data['Noise_Count'],
            'Silhouette_Score': data['Silhouette'],
            'Cohesion': data['Cohesion'],
            'Tiempo_Ejecucion_Segs': data['Time_Secs']
        })
    df_bench = pd.DataFrame(rows)
    df_bench.to_csv(RUN_DIR / 'clustering_metrics.csv', index=False)
    
    logger.info("\n--- MÉTRICAS DE CLUSTERING ---")
    logger.info(f"\n{df_bench.to_string(index=False)}")
    return df_bench

if __name__ == '__main__':
    execute_clustering_pipeline()
