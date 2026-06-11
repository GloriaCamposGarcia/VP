import time
import torch
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from networkx.algorithms.community import louvain_communities

from src.config import logger, DATA_PROCESSED_DIR

def load_graph_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga el listado de entidades y las aristas reales del grafo desde la carpeta de datos procesados.
    """
    nodes_path = DATA_PROCESSED_DIR / 'consolidated_entities.csv'
    edges_path = DATA_PROCESSED_DIR / 'entity_edges.csv'
    
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError("Faltan archivos procesados. Ejecute el script de procesamiento primero.")
        
    df_nodes = pd.read_csv(nodes_path)
    df_edges = pd.read_csv(edges_path)
    logger.info(f"Datos de grafos cargados: Nodos={len(df_nodes)}, Aristas={len(df_edges)}")
    return df_nodes, df_edges

def perform_topological_analysis(
    df_nodes: pd.DataFrame, 
    df_edges: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Realiza un análisis topológico de la red relacional utilizando NetworkX.
    Calcula PageRank, Grado de Centralidad y detecta comunidades usando Louvain.
    Retorna los nodos enriquecidos y la tabla de resumen de las comunidades detectadas.
    """
    logger.info("Iniciando análisis topológico con NetworkX...")
    t0 = time.time()
    
    # 1. Construir el grafo de NetworkX
    G = nx.Graph()
    for _, row in df_nodes.iterrows():
        G.add_node(
            row['entity_id'], 
            name=row['entity_name'], 
            type=row['entity_type'],
            country=row['country_code']
        )
        
    for _, row in df_edges.iterrows():
        G.add_edge(row['source'], row['target'], weight=float(row['weight']))
        
    # 2. Calcular PageRank y Grado de Centralidad
    logger.info("Calculando métricas de centralidad (PageRank y Grado)...")
    pagerank_scores = nx.pagerank(G, weight='weight')
    degree_scores = dict(G.degree())
    
    # 3. Detectar comunidades Louvain (agrupación estructural de entidades conectadas)
    logger.info("Agrupando entidades en comunidades mediante algoritmo Louvain...")
    communities = louvain_communities(G, weight='weight', seed=42)
    
    # Mapear cada nodo a su ID de comunidad
    node_to_comm = {}
    for comm_idx, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = comm_idx
            
    # 4. Integrar métricas a la tabla de entidades
    df_nodes_enriched = df_nodes.copy()
    df_nodes_enriched['pagerank'] = df_nodes_enriched['entity_id'].map(pagerank_scores).fillna(0.0)
    df_nodes_enriched['degree'] = df_nodes_enriched['entity_id'].map(degree_scores).fillna(0).astype(int)
    df_nodes_enriched['community_id'] = df_nodes_enriched['entity_id'].map(node_to_comm).fillna(-1).astype(int)
    
    # 5. Generar resumen de comunidades
    comm_summary = []
    for comm_idx, comm in enumerate(communities):
        comm_nodes = list(comm)
        size = len(comm_nodes)
        
        # Obtener los miembros principales (primeros 5)
        names = [G.nodes[n].get('name', n) for n in comm_nodes[:5]]
        
        comm_summary.append({
            'community_id': comm_idx,
            'size': size,
            'miembros_principales': ", ".join(names) + ("..." if size > 5 else "")
        })
        
    df_comm = pd.DataFrame(comm_summary).sort_values(by='size', ascending=False).reset_index(drop=True)
    
    elapsed = time.time() - t0
    logger.info(f"Análisis topológico completado en {elapsed:.2f}s. Comunidades encontradas: {len(communities)}")
    return df_nodes_enriched, df_comm

def propagate_features_gnn(
    df_nodes: pd.DataFrame, 
    df_edges: pd.DataFrame
) -> np.ndarray:
    """
    Aplica una capa de convolución de grafos (GCN) de PyTorch Geometric para propagar
    y suavizar (smooth) las características de comportamiento de los nodos a través
    de las aristas de relaciones reales, capturando patrones estructurales locales.
    """
    logger.info("Propagando características a través del grafo usando PyTorch Geometric...")
    t0 = time.time()
    
    # 1. Crear mapeo de ID de entidad a índice entero
    node_ids = df_nodes['entity_id'].tolist()
    entity_to_idx = {uid: idx for idx, uid in enumerate(node_ids)}
    
    # 2. Preparar matriz de características de entrada X
    feature_cols = ['sources_evaluated', 'sources_with_hallazgo', 'max_identity_score', 'evidence_items']
    X_raw = df_nodes[feature_cols].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    x_tensor = torch.tensor(X_scaled, dtype=torch.float)
    
    # 3. Construir índice de aristas
    valid_edges = df_edges[
        df_edges['source'].isin(entity_to_idx) & df_edges['target'].isin(entity_to_idx)
    ].copy()
    
    source_idx = valid_edges['source'].map(entity_to_idx).values
    target_idx = valid_edges['target'].map(entity_to_idx).values
    
    edges_src = np.concatenate([source_idx, target_idx])
    edges_tgt = np.concatenate([target_idx, source_idx])
    edge_index = torch.tensor(np.stack([edges_src, edges_tgt]), dtype=torch.long)
    
    # Cargar pesos si existen
    weights = valid_edges['weight'].values
    edge_weight = torch.tensor(np.concatenate([weights, weights]), dtype=torch.float) if len(weights) > 0 else None
    
    # 4. Inicializar y aplicar capa de Convolución GCN
    # Se utiliza una capa GCNConv sin entrenamiento para suavizar características
    # agregando información del vecindario relacional.
    in_features = len(feature_cols)
    out_features = 4  # Dimensión del embedding relacional de salida
    
    conv = GCNConv(in_channels=in_features, out_channels=out_features)
    
    # Fijar pesos uniformes para asegurar consistencia determinista en la propagación
    with torch.no_grad():
        conv.lin.weight.fill_(0.5)
        if conv.bias is not None:
            conv.bias.fill_(0.0)
            
    # Propagación
    with torch.no_grad():
        H = conv(x_tensor, edge_index, edge_weight)
        H_numpy = H.numpy()
        
    elapsed = time.time() - t0
    logger.info(f"Propagación GNN completada en {elapsed:.2f}s. Dimensión de salida: {H_numpy.shape}")
    return H_numpy

def execute_unsupervised_graph_analysis() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orquesta el pipeline completo de análisis de grafos no supervisado y guarda resultados.
    """
    df_nodes, df_edges = load_graph_inputs()
    
    # 1. Análisis topológico
    df_nodes_enriched, df_comm = perform_topological_analysis(df_nodes, df_edges)
    
    # 2. Propagación GNN de características
    H_relational = propagate_features_gnn(df_nodes_enriched, df_edges)
    
    # Guardar las características con el contexto relacional agregado
    for i in range(H_relational.shape[1]):
        df_nodes_enriched[f'gnn_feature_{i}'] = H_relational[:, i]
        
    # Guardar los datasets enriquecidos en processed/
    df_nodes_enriched.to_csv(DATA_PROCESSED_DIR / 'graph_enriched_entities.csv', index=False)
    df_comm.to_csv(DATA_PROCESSED_DIR / 'graph_communities.csv', index=False)
    
    logger.info("Pipeline de grafos no supervisado completado exitosamente.")
    logger.info(f"Top 5 comunidades por tamaño:\n{df_comm.head(5).to_string(index=False)}")
    
    return df_nodes_enriched, df_comm

if __name__ == '__main__':
    execute_unsupervised_graph_analysis()
