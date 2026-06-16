import os
import sys
import time
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from src.config import logger, RUN_DIR, RUN_DATE

def plot_single_entity_graph(args):
    """
    Función de trabajador para generar e ilustrar el minigrafo ego de 1 salto para una entidad.
    Se importan las bibliotecas de dibujo dentro de la función para evitar conflictos
    de hilos y serialización en Windows.
    """
    ent_id, ent_name, connections, output_dir_str, name_map = args
    
    # Priorizar conexiones semánticas ocultas, luego ordenar por peso descendente
    connections = sorted(
        connections,
        key=lambda x: (x[1] == 'semantic_similarity', x[2]),
        reverse=True
    )
    # Limitar a un máximo de 30 conexiones para legibilidad del grafo
    max_conn = 30
    if len(connections) > max_conn:
        connections = connections[:max_conn]
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    G = nx.Graph()
    # Agregar nodo central (dorado)
    G.add_node(ent_id, name=ent_name, is_center=True)
    
    for neighbor_id, rel, w, is_source in connections:
        neigh_name = name_map.get(neighbor_id, neighbor_id)
        G.add_node(neighbor_id, name=neigh_name, is_center=False)
        G.add_edge(ent_id, neighbor_id, relation=rel, weight=w)
        
    # Layout concéntrico: Entidad principal en el centro (0,0), vecinos en la circunferencia
    pos = {ent_id: np.array([0.0, 0.0])}
    neighbors = [n for n in G.nodes if n != ent_id]
    n_neigh = len(neighbors)
    
    for idx, neigh in enumerate(neighbors):
        angle = 2.0 * np.pi * idx / n_neigh
        pos[neigh] = np.array([np.cos(angle), np.sin(angle)])
        
    # Dibujar la figura
    fig = plt.figure(figsize=(7, 7), facecolor='#1E1E1E')
    ax = plt.gca()
    ax.set_facecolor('#1E1E1E')
    
    # Estilos de nodos
    center_nodes = [n for n, d in G.nodes(data=True) if d.get('is_center', False)]
    neigh_nodes = [n for n, d in G.nodes(data=True) if not d.get('is_center', False)]
    
    # Centro en color dorado brillante, vecinos en azul cielo
    nx.draw_networkx_nodes(G, pos, nodelist=center_nodes, node_size=900, node_color='#FFD700', edgecolors='#2D2D2D', linewidths=1.5, alpha=0.95)
    nx.draw_networkx_nodes(G, pos, nodelist=neigh_nodes, node_size=350, node_color='#3498DB', edgecolors='#2D2D2D', linewidths=1.2, alpha=0.85)
    
    # Separar aristas físicas y semánticas
    edges_phys = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') != 'semantic_similarity']
    edges_sem = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'semantic_similarity']
    
    # Enlaces físicos: gris continuo y delgado
    nx.draw_networkx_edges(G, pos, edgelist=edges_phys, width=1.2, edge_color='#5A5A5A', alpha=0.5)
    # Enlaces semánticos (ocultos): rojo discontinuo grueso
    nx.draw_networkx_edges(G, pos, edgelist=edges_sem, width=2.5, edge_color='#FF4C4C', alpha=0.85, style='dashed')
    
    # Etiquetas de texto
    labels = {n: G.nodes[n].get('name', n) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='#FFFFFF', font_family='sans-serif',
                           bbox=dict(facecolor='#1E1E1E', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
    
    plt.title(f"Red Ego de Relaciones: {ent_name}\n({ent_id}) | Fecha: {RUN_DATE}", color='#FFFFFF', fontsize=11, fontweight='bold', pad=15)
    plt.axis('off')
    
    # Leyenda compacta
    custom_legend = [
        Line2D([0], [0], color='#5A5A5A', alpha=0.7, lw=1.2, label='Enlace Físico (URL/Hash)'),
        Line2D([0], [0], color='#FF4C4C', lw=2.0, linestyle='--', label='Vínculo Semántico Oculto'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700', markersize=8, linestyle='none', label='Entidad Consultada'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB', markersize=6, linestyle='none', label='Entidad Vinculada')
    ]
    plt.legend(handles=custom_legend, loc='lower center', facecolor='#2D2D2D', edgecolor='none', labelcolor='#FFFFFF', fontsize=8, ncol=2)
    
    out_path = Path(output_dir_str) / f"{ent_id}_{RUN_DATE}.png"
    plt.savefig(out_path, dpi=120, facecolor='#1E1E1E', bbox_inches='tight')
    plt.close(fig)

def generate_all_entity_graphs():
    """
    Orquesta la generación masiva de grafos individuales utilizando multiprocessing.
    """
    logger.info("Iniciando generación masiva de grafos individuales por Entity ID...")
    
    nodes_path = RUN_DIR / 'graph_enriched_entities.csv'
    if not nodes_path.exists():
        nodes_path = RUN_DIR / 'consolidated_entities.csv'
        
    edges_path = RUN_DIR / 'entity_edges.csv'
    
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError("No se encontraron los datos del grafo. Ejecute el pipeline primero.")
        
    df_nodes = pd.read_csv(nodes_path)
    df_edges = pd.read_csv(edges_path)
    
    # Crear el directorio extra
    output_dir = RUN_DIR / 'entity_graphs'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorio de visualizaciones individuales verificado en: {output_dir}")
    
    name_map = dict(zip(df_nodes['entity_id'], df_nodes['entity_name']))
    
    # Agrupar aristas por nodo para evitar búsquedas lentas (optimizado con zip en vez de iterrows)
    from collections import defaultdict
    adj = defaultdict(list)
    for s, t, rel, w in zip(df_edges['source'].values, df_edges['target'].values, df_edges['relation_type'].values, df_edges['weight'].values):
        adj[s].append((t, rel, w, True))
        adj[t].append((s, rel, w, False))
        
    # Recopilar tareas para el Pool
    tasks = []
    # Ordenar nodos para priorizar entidades sospechosas, en lista negra o con mayor centralidad PageRank
    df_nodes_sorted = df_nodes.copy()
    sort_cols = []
    if 'is_suspicious_analyst' in df_nodes_sorted.columns:
        sort_cols.append('is_suspicious_analyst')
    if 'pagerank' in df_nodes_sorted.columns:
        sort_cols.append('pagerank')
        
    if sort_cols:
        df_nodes_sorted = df_nodes_sorted.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
        
    for ent_id in df_nodes_sorted['entity_id'].dropna().unique():
        connections = adj[ent_id]
        if connections:
            name = name_map.get(ent_id, ent_id)
            tasks.append((ent_id, name, connections, str(output_dir), name_map))
            
    # Limitar la generación a un máximo de 10 redes de ego críticas para optimizar rendimiento y almacenamiento
    max_graphs = 10
    if len(tasks) > max_graphs:
        logger.info(f"Limitando visualizaciones de {len(tasks)} a las {max_graphs} más críticas para optimizar rendimiento.")
        tasks = tasks[:max_graphs]
            
    logger.info(f"Total de entidades conectadas a dibujar: {len(tasks)}")
    
    # Multiprocesamiento
    import multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Inicializando Multiprocessing Pool con {num_workers} cores...")
    
    t0 = time.time()
    with multiprocessing.Pool(processes=num_workers) as pool:
        count = 0
        for _ in pool.imap_unordered(plot_single_entity_graph, tasks, chunksize=25):
            count += 1
            if count % 1000 == 0:
                logger.info(f"Progreso: {count}/{len(tasks)} imágenes de red generadas...")
                
    elapsed = time.time() - t0
    logger.info(f"Visualización masiva completada en {elapsed:.2f}s (Promedio: {elapsed/len(tasks):.4f}s por imagen).")

if __name__ == '__main__':
    generate_all_entity_graphs()
