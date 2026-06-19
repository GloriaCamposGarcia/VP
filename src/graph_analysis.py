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

from src.config import logger, RUN_DIR

def load_graph_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Se realiza la carga del catálogo de entidades consolidadas y las aristas desde el directorio de la corrida.
    """
    nodes_path = RUN_DIR / 'consolidated_entities.csv'
    edges_path = RUN_DIR / 'entity_edges.csv'
    
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError("Faltan archivos procesados. Ejecute el script de procesamiento primero.")
        
    df_nodes = pd.read_csv(nodes_path)
    df_edges = pd.read_csv(edges_path)
    logger.info(f"Datos del grafo cargados (nodos: {len(df_nodes)}, aristas: {len(df_edges)}).")
    return df_nodes, df_edges

def perform_topological_analysis(
    df_nodes: pd.DataFrame, 
    df_edges: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Se realiza un análisis topológico de la red relacional mediante la librería NetworkX.
    Se determinan las métricas de PageRank, grado de centralidad y comunidades Louvain.
    """
    logger.info("Análisis topológico de red en progreso.")
    t0 = time.time()
    
    # 1. Consturcción del grafo mediante la estructura nx.Graph
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
        
    # 2. Métricas de PageRank y centralidad de grado
    logger.info("Cálculo de PageRank y centralidad de grado en progreso.")
    pagerank_scores = nx.pagerank(G, weight='weight')
    degree_scores = dict(G.degree())
    
    # 3. Detección de las comunidades Louvain
    logger.info("Detección de comunidades Louvain en progreso.")
    communities = louvain_communities(G, weight='weight', seed=42)
    
    # Se asocia cada nodo a su correspondiente identificador de comunidad
    node_to_comm = {}
    for comm_idx, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = comm_idx
            
    # 4. Integración de las métricas en la tabla de entidades
    df_nodes_enriched = df_nodes.copy()
    df_nodes_enriched['pagerank'] = df_nodes_enriched['entity_id'].map(pagerank_scores).fillna(0.0)
    df_nodes_enriched['degree'] = df_nodes_enriched['entity_id'].map(degree_scores).fillna(0).astype(int)
    df_nodes_enriched['community_id'] = df_nodes_enriched['entity_id'].map(node_to_comm).fillna(-1).astype(int)
    
    # 5. Tabla resumen de comunidades Louvain
    comm_summary = []
    for comm_idx, comm in enumerate(communities):
        comm_nodes = list(comm)
        size = len(comm_nodes)
        
        # Obtención de los nombres de los miembros principales de la comunidad
        names = [G.nodes[n].get('name', n) for n in comm_nodes[:5]]
        
        comm_summary.append({
            'community_id': comm_idx,
            'size': size,
            'miembros_principales': ", ".join(names) + ("..." if size > 5 else "")
        })
        
    df_comm = pd.DataFrame(comm_summary).sort_values(by='size', ascending=False).reset_index(drop=True)
    
    elapsed = time.time() - t0
    logger.info(f"Análisis finalizado en {elapsed:.2f}s (comunidades: {len(communities)}).")
    return df_nodes_enriched, df_comm

def propagate_features_gnn(
    df_nodes: pd.DataFrame, 
    df_edges: pd.DataFrame
) -> np.ndarray:
    """
    Se aplica una capa de convolución de grafos (GCN) de PyTorch Geometric para propagar y suavizar
    las características de comportamiento a través de la topología relacional del grafo.
    """
    logger.info("Propagación de características relacionales vía GNN en progreso.")
    t0 = time.time()
    
    # 1. Mapeo de identificador a índice entero
    node_ids = df_nodes['entity_id'].tolist()
    entity_to_idx = {uid: idx for idx, uid in enumerate(node_ids)}
    
    # 2. Preparación y escala de la matriz de características de entrada
    feature_cols = ['sources_evaluated', 'sources_with_hallazgo', 'max_identity_score', 'evidence_items']
    X_raw = df_nodes[feature_cols].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    x_tensor = torch.tensor(X_scaled, dtype=torch.float)
    
    # 3. Construcción del índice tensor de aristas (edge_index)
    valid_edges = df_edges[
        df_edges['source'].isin(entity_to_idx) & df_edges['target'].isin(entity_to_idx)
    ].copy()
    
    source_idx = valid_edges['source'].map(entity_to_idx).values
    target_idx = valid_edges['target'].map(entity_to_idx).values
    
    edges_src = np.concatenate([source_idx, target_idx])
    edges_tgt = np.concatenate([target_idx, source_idx])
    edge_index = torch.tensor(np.stack([edges_src, edges_tgt]), dtype=torch.long)
    
    # Se cargan y transforman los pesos de las aristas
    weights = valid_edges['weight'].values
    edge_weight = torch.tensor(np.concatenate([weights, weights]), dtype=torch.float) if len(weights) > 0 else None
    
    # 4. Capa de convolución GCN
    in_features = len(feature_cols)
    out_features = 4
    
    conv = GCNConv(in_channels=in_features, out_channels=out_features)
    
    # Se fijan los pesos a valores uniformes para garantizar reproducibilidad
    with torch.no_grad():
        conv.lin.weight.fill_(0.5)
        if conv.bias is not None:
            conv.bias.fill_(0.0)
            
    # Se ejecuta la propagación sin gradiente
    with torch.no_grad():
        H = conv(x_tensor, edge_index, edge_weight)
        H_numpy = H.numpy()
        
    elapsed = time.time() - t0
    logger.info(f"Propagación GNN finalizada en {elapsed:.2f}s (dimensión: {H_numpy.shape}).")
    return H_numpy

def execute_unsupervised_graph_analysis() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Se ejecuta el pipeline completo de análisis de grafos no supervisado, persistiendo los resultados enriquecidos.
    """
    df_nodes, df_edges = load_graph_inputs()
    
    # 1. Análisis topológico
    df_nodes_enriched, df_comm = perform_topological_analysis(df_nodes, df_edges)
    
    # 2. Propagación GNN de características
    H_relational = propagate_features_gnn(df_nodes_enriched, df_edges)
    
    # Se agregan las características resultantes del modelo GCN
    for i in range(H_relational.shape[1]):
        df_nodes_enriched[f'gnn_feature_{i}'] = H_relational[:, i]
        
    # Se guardan los conjuntos de datos enriquecidos en formato CSV
    df_nodes_enriched.to_csv(RUN_DIR / 'graph_enriched_entities.csv', index=False)
    df_comm.to_csv(RUN_DIR / 'graph_communities.csv', index=False)
    
    logger.info("Pipeline de grafos completado.")
    logger.info(f"Top 5 comunidades Louvain:\n{df_comm.head(5).to_string(index=False)}")
    
    return df_nodes_enriched, df_comm

if __name__ == '__main__':
    execute_unsupervised_graph_analysis()
