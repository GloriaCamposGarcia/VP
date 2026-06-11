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

from src.config import logger, DATA_PROCESSED_DIR, HF_EMBEDDING_MODEL, EMBEDDING_BACKEND

def generate_embeddings(texts: pd.Series) -> Tuple[np.ndarray, str]:
    """
    Genera embeddings para la lista de textos utilizando SentenceTransformers o TF-IDF.
    Implementa caché en disco para evitar re-cálculos costosos.
    """
    cache_path = DATA_PROCESSED_DIR / 'entity_embeddings_cache.npy'
    
    if cache_path.exists():
        logger.info(f"Cargando embeddings desde el caché: {cache_path}")
        X = np.load(cache_path)
        return X, EMBEDDING_BACKEND

    logger.info(f"Generando embeddings nuevos usando backend: {EMBEDDING_BACKEND}...")
    texts_list = texts.fillna("").astype(str).tolist()

    if EMBEDDING_BACKEND in ['sentence-transformers', 'hf', 'sentencetransformers']:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Inicializando modelo local SentenceTransformer: {HF_EMBEDDING_MODEL}")
            model = SentenceTransformer(HF_EMBEDDING_MODEL)
            # Codificar
            embeddings = model.encode(texts_list, show_progress_bar=True, convert_to_numpy=True)
            # Normalizar L2 para similitud coseno equivalente a distancia euclidiana
            X = normalize(embeddings, norm='l2', axis=1)
            np.save(cache_path, X)
            logger.info("Embeddings calculados y guardados en caché.")
            return X, 'sentence-transformers'
        except Exception as e:
            logger.error(f"Error al cargar SentenceTransformers: {e}. Usando TF-IDF como fallback...")

    # Fallback a TF-IDF
    logger.info("Calculando representación de texto usando TF-IDF...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char_wb', min_df=2)
    X_sparse = vectorizer.fit_transform(texts_list)
    X = normalize(X_sparse.toarray(), norm='l2', axis=1)
    np.save(cache_path, X)
    logger.info("Embeddings TF-IDF calculados y guardados en caché.")
    return X, 'tfidf'

def reduce_dimensions(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica PCA y t-SNE para reducir la dimensionalidad de los embeddings.
    """
    logger.info("Aplicando reducción de dimensionalidad PCA (2 componentes)...")
    t0 = time.time()
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    logger.info(f"PCA completado en {time.time() - t0:.2f}s.")

    logger.info("Aplicando reducción de dimensionalidad t-SNE (2 componentes)...")
    t0 = time.time()
    # Si la cantidad de muestras es muy baja para la perplejidad por defecto (30), se ajusta automáticamente.
    n_samples = X.shape[0]
    perp = min(30, max(5, n_samples - 1))
    tsne = TSNE(n_components=2, random_state=42, max_iter=300, perplexity=perp, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)
    logger.info(f"t-SNE completado en {time.time() - t0:.2f}s con perplejidad {perp}.")

    return X_pca, X_tsne

def calculate_cohesion(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calcula la cohesión del clúster (promedio de las distancias euclidianas 
    de cada punto a su respectivo centroide).
    Valores menores representan clústeres más cohesivos.
    """
    unique_labels = np.unique(labels)
    # Excluir etiqueta de ruido (-1) si existe
    unique_labels = [l for l in unique_labels if l != -1]
    
    if not unique_labels:
        return np.nan
        
    distances = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = cluster_points.mean(axis=0)
        # Distancia euclidiana de cada punto al centroide del clúster
        dist = np.linalg.norm(cluster_points - centroid, axis=1)
        distances.extend(dist.tolist())
        
    return float(np.mean(distances))

def run_clustering_benchmarking(X: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Ejecuta K-Means y HDBSCAN, evalúa su desempeño y calidad de clústeres.
    """
    logger.info("Iniciando Benchmarking de Clustering...")
    results = {}
    cluster_labels = {}

    n_samples = X.shape[0]
    n_clusters_kmeans = min(5, n_samples) if n_samples > 1 else 1

    # 1. K-Means
    logger.info(f"Entrenando K-Means con {n_clusters_kmeans} clústeres...")
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
    logger.info("Entrenando HDBSCAN...")
    t0 = time.time()
    # min_cluster_size y min_samples dinámicos para evitar errores con datasets pequeños
    min_cluster = min(10, max(2, n_samples // 3))
    min_samp = min(5, max(1, min_cluster // 2))
    
    logger.info(f"HDBSCAN min_cluster_size={min_cluster}, min_samples={min_samp}")
    hdb = HDBSCAN(min_cluster_size=min_cluster, min_samples=min_samp, metric='euclidean')
    hdb_labels = hdb.fit_predict(X)
    hdb_time = time.time() - t0
    cluster_labels['HDBSCAN'] = hdb_labels

    # Métricas HDBSCAN
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
    Ejecuta el pipeline completo de clustering semántico a nivel de entidad,
    detecta vínculos AML ocultos mediante similitud de coseno y
    calcula anomalías en su información de forma no supervisada.
    """
    entities_path = DATA_PROCESSED_DIR / 'consolidated_entities.csv'
    evidence_path = DATA_PROCESSED_DIR / 'processed_evidence_items.csv'
    
    if not entities_path.exists():
        raise FileNotFoundError(f"No se encontró {entities_path}. Ejecuta primero data_processing.py")
    if not evidence_path.exists():
        raise FileNotFoundError(f"No se encontró {evidence_path}. Ejecuta primero data_processing.py")

    df_entities = pd.read_csv(entities_path)
    df_evidence = pd.read_csv(evidence_path)
    
    logger.info("Agrupando evidencias a nivel de entidad para construir perfiles de texto...")
    df_evidence['snippet_clean'] = df_evidence['snippet'].fillna('').astype(str).str.strip()
    df_evidence['reason_clean'] = df_evidence['review_reason'].fillna('').astype(str).str.strip()
    
    # Combinar fragmento y contexto para cada ítem de evidencia
    df_evidence['item_text'] = df_evidence.apply(
        lambda r: f"Fragmento: {r['snippet_clean']} | Motivo: {r['reason_clean']}" 
        if r['reason_clean'] else f"Fragmento: {r['snippet_clean']}", 
        axis=1
    )
    
    # Agrupar por entity_id y unir
    grouped_evidence = df_evidence.groupby('entity_id')['item_text'].apply(
        lambda x: " ; ".join(x.unique())
    ).rename('all_evidence')
    
    # Combinar con catálogo de entidades
    # Eliminar columna predominant_cluster anterior si ya existe para evitar duplicación
    if 'predominant_cluster' in df_entities.columns:
        df_entities = df_entities.drop(columns=['predominant_cluster'])
    df_entities = df_entities.merge(grouped_evidence, on='entity_id', how='left')
    df_entities['all_evidence'] = df_entities['all_evidence'].fillna("Sin evidencias recopiladas").replace("", "Sin evidencias recopiladas")
    
    # Construir el perfil de texto descriptivo de cada entidad
    df_entities['entity_profile_text'] = df_entities.apply(
        lambda r: f"Entidad: {r['entity_name']} | Tipo: {r['entity_type']} | País: {r['country_code']} | Evidencias: {r['all_evidence']}",
        axis=1
    )
    
    # Asegurar que no haya vacíos
    df_entities['entity_profile_text'] = df_entities['entity_profile_text'].fillna("Sin perfil disponible")

    # Obtener embeddings a nivel de perfil de entidad
    X, backend = generate_embeddings(df_entities['entity_profile_text'])
    
    # Ejecutar clustering sobre los embeddings de entidades
    metrics, labels = run_clustering_benchmarking(X)
    
    # Reducción dimensional para visualizaciones
    X_pca, X_tsne = reduce_dimensions(X)
    
    # Guardar etiquetas de clústeres en los datos de entidades
    df_entities['kmeans_cluster'] = labels['K-Means']
    df_entities['hdbscan_cluster'] = labels['HDBSCAN']
    df_entities['predominant_cluster'] = labels['K-Means'] # Mantener por compatibilidad
    
    # Guardar coordenadas dimensionales
    df_entities['pca_x'] = X_pca[:, 0]
    df_entities['pca_y'] = X_pca[:, 1]
    df_entities['tsne_x'] = X_tsne[:, 0]
    df_entities['tsne_y'] = X_tsne[:, 1]
    
    # Asignar etiquetas interpretativas basadas en K-Means (5 clústeres)
    cluster_themes = {
        0: "Sanciones Financieras e Internacionales",
        1: "Personas de Alto Riesgo / PEPs",
        2: "SAT Listas Negras / Discrepancias Fiscales",
        3: "Listas de Alerta Locales / Medios Adversos",
        4: "Registros de Control General / Homónimos"
    }
    df_entities['kmeans_cluster_theme'] = df_entities['kmeans_cluster'].map(cluster_themes).fillna("Otros Hallazgos")
    
    # 1. DETECCIÓN DE VÍNCULOS AML OCULTOS (Similitud Coseno)
    logger.info("Detectando vínculos AML ocultos entre entidades por similitud de embeddings...")
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(X)
    
    links = []
    n_entities = len(df_entities)
    for i in range(n_entities):
        for j in range(i + 1, n_entities):
            sim = float(similarity_matrix[i, j])
            if sim >= 0.70: # Umbral de similitud semántica alta
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
        links_output_path = DATA_PROCESSED_DIR / 'hidden_entity_links.csv'
        df_links.to_csv(links_output_path, index=False)
        logger.info(f"Vínculos semánticos ocultos guardados en: {links_output_path} (Total: {len(df_links)})")
        
        # Inyectar estos vínculos en el grafo relacional (entity_edges.csv)
        edges_path = DATA_PROCESSED_DIR / 'entity_edges.csv'
        if edges_path.exists():
            df_edges = pd.read_csv(edges_path)
        else:
            df_edges = pd.DataFrame(columns=['source', 'target', 'relation_type', 'weight'])
            
        df_links_for_merge = df_links[['source', 'target', 'relation_type', 'weight']]
        df_edges_combined = pd.concat([df_edges, df_links_for_merge], ignore_index=True)
        # Consolidar duplicados tomando el peso máximo
        df_edges_combined = df_edges_combined.groupby(['source', 'target', 'relation_type'], as_index=False)['weight'].max()
        df_edges_combined.to_csv(edges_path, index=False)
        logger.info(f"Se inyectaron y consolidaron {len(df_links)} aristas de similitud semántica en {edges_path}")
    else:
        logger.info("No se encontraron vínculos de alta similitud semántica (>=0.70).")
        # Asegurar archivo vacío pero con columnas correctas
        pd.DataFrame(columns=['source', 'source_name', 'target', 'target_name', 'relation_type', 'weight']).to_csv(
            DATA_PROCESSED_DIR / 'hidden_entity_links.csv', index=False
        )

    # 2. DETECCIÓN DE ANOMALÍAS EN LA INFORMACIÓN (No Supervisado)
    logger.info("Calculando indicadores de anomalías no supervisadas sobre perfiles de entidades...")
    from sklearn.ensemble import IsolationForest
    
    # Isolation Forest sobre embeddings
    iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
    iso_forest.fit(X)
    
    # El score de anomalía (mayor es más inusual)
    raw_scores = iso_forest.score_samples(X)
    df_entities['anomaly_score_embedding'] = -raw_scores
    df_entities['is_embedding_outlier'] = np.where(iso_forest.predict(X) == -1, 1, 0)
    
    # Calcular distancia al centroide del clúster K-Means correspondiente
    # Primero obtenemos el objeto KMeans para calcular distancias
    n_clusters_kmeans = min(5, n_entities) if n_entities > 1 else 1
    if n_entities > 1:
        kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
        kmeans.fit(X)
        distances = kmeans.transform(X) # Distancia a todos los centroides
        assigned_clusters = labels['K-Means']
        dist_to_centroid = np.array([distances[idx, cluster] for idx, cluster in enumerate(assigned_clusters)])
        df_entities['distance_to_centroid'] = dist_to_centroid
    else:
        df_entities['distance_to_centroid'] = 0.0

    # Guardar catálogo consolidado de entidades actualizado
    df_entities.to_csv(DATA_PROCESSED_DIR / 'consolidated_entities.csv', index=False)
    logger.info(f"Catálogo consolidado de entidades actualizado en: {DATA_PROCESSED_DIR / 'consolidated_entities.csv'}")

    # Generar tabla comparativa en dataframe
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
    df_bench.to_csv(DATA_PROCESSED_DIR / 'clustering_metrics.csv', index=False)
    
    logger.info("\n--- METRICAS DE CLUSTERING DE PERFILES DE ENTIDAD ---")
    logger.info(f"\n{df_bench.to_string(index=False)}")
    return df_bench

if __name__ == '__main__':
    execute_clustering_pipeline()
