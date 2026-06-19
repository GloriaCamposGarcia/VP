import os
import sys
import time
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from src.config import logger, RUN_DIR, RUN_DATE, DATA_RAW_DIR

def get_entity_evidence_details(entity_id, df_evidence):
    """
    Busca en df_evidence las coincidencias para entity_id y devuelve un diccionario
    con detalles de las listas y motivos de sanción.
    """
    if df_evidence.empty or 'entity_id' not in df_evidence.columns:
        return None
        
    sub_ev = df_evidence[df_evidence['entity_id'] == entity_id]
    if sub_ev.empty:
        return None
        
    details = {
        'lists': [],
        'reasons': [],
        'status': None,
        'country': None
    }
    
    for _, row in sub_ev.iterrows():
        source = str(row.get('source_id', '')).strip()
        if source and source not in details['lists']:
            details['lists'].append(source)
            
        country = str(row.get('country', '')).strip()
        if country and not details['country']:
            details['country'] = country
            
        # Extraer de extracted_fields
        ext_fields_str = row.get('extracted_fields', '')
        if pd.notna(ext_fields_str) and str(ext_fields_str).strip():
            try:
                import ast
                fields = ast.literal_eval(str(ext_fields_str))
                if isinstance(fields, dict):
                    reason = fields.get('statement_of_reasons', '') or fields.get('csd_cancellation_reason', '') or fields.get('review_reason', '')
                    if reason and reason not in details['reasons']:
                        details['reasons'].append(reason)
                        
                    regime = fields.get('regime_name', '')
                    if regime and f"Regime: {regime}" not in details['reasons']:
                        details['reasons'].append(f"Regime: {regime}")
                        
                    programs = fields.get('programs_json', '')
                    if programs:
                        if isinstance(programs, str) and (programs.startswith('[') or programs.startswith('{')):
                            import json
                            try:
                                clean_prog = programs.replace("'", '"')
                                prog_list = json.loads(clean_prog)
                                if isinstance(prog_list, list):
                                    for p in prog_list:
                                        if f"Prog: {p}" not in details['reasons']:
                                            details['reasons'].append(f"Prog: {p}")
                            except Exception:
                                pass
                        elif f"Prog: {programs}" not in details['reasons']:
                            details['reasons'].append(f"Prog: {programs}")
                            
                    status = fields.get('status', '') or fields.get('supposition', '')
                    if status and not details['status']:
                        details['status'] = status
            except Exception:
                pass
                
        # Fallback de motivo en snippet
        snippet = str(row.get('snippet', '')).strip()
        if snippet and not details['reasons']:
            details['reasons'].append(snippet)
            
    return details

def plot_single_entity_graph(args):
    """
    Se genera y exporta el minigrafo ego de 1 salto para una entidad específica.
    Se importan las bibliotecas gráficas internamente para prevenir problemas de concurrencia y serialización en Windows.
    """
    ent_id, ent_name, connections, output_dir_str, name_map = args
    
    # Se priorizan las conexiones semánticas y se ordenan por peso descendentemente
    connections = sorted(
        connections,
        key=lambda x: (x[1] == 'semantic_similarity', x[2]),
        reverse=True
    )
    # Se restringe a un máximo de 30 conexiones por legibilidad
    max_conn = 30
    if len(connections) > max_conn:
        connections = connections[:max_conn]
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import textwrap
    
    # Cargar df_evidence
    evidence_path = DATA_RAW_DIR / "evidence_items.csv"
    if evidence_path.exists():
        df_evidence = pd.read_csv(evidence_path)
    else:
        df_evidence = pd.DataFrame()
        
    # Cargar mapa de anomalías de consolidated_entities
    outlier_map = {}
    consolidated_path = RUN_DIR / 'consolidated_entities.csv'
    if consolidated_path.exists():
        try:
            df_nodes_temp = pd.read_csv(consolidated_path)
            df_nodes_temp['is_outlier'] = (
                (df_nodes_temp.get('anomaly_isolationforest', 0) == 1) |
                (df_nodes_temp.get('anomaly_oneclasssvm', 0) == 1) |
                (df_nodes_temp.get('anomaly_localoutlierfactor', 0) == 1) |
                (df_nodes_temp.get('is_embedding_outlier', 0) == 1)
            ).astype(int)
            outlier_map = dict(zip(df_nodes_temp['entity_id'], df_nodes_temp['is_outlier']))
        except Exception:
            pass
            
    G = nx.Graph()
    # Se agrega el nodo central en color dorado
    G.add_node(ent_id, name=ent_name, is_center=True)
    
    for neighbor_id, rel, w, is_source in connections:
        neigh_name = name_map.get(neighbor_id, neighbor_id)
        G.add_node(neighbor_id, name=neigh_name, is_center=False)
        G.add_edge(ent_id, neighbor_id, relation=rel, weight=w)
        
    # Se define un layout concéntrico ubicando el centro en (0,0) y los vecinos en la circunferencia
    pos = {ent_id: np.array([0.0, 0.0])}
    neighbors = [n for n in G.nodes if n != ent_id]
    n_neigh = len(neighbors)
    
    for idx, neigh in enumerate(neighbors):
        angle = 2.0 * np.pi * idx / n_neigh
        pos[neigh] = np.array([np.cos(angle) * 2.0, np.sin(angle) * 2.0])
        
    # Se genera el lienzo de la figura con fondo oscuro y mayor tamaño para acomodar las etiquetas detalladas
    fig = plt.figure(figsize=(9.5, 9.5), facecolor='#1E1E1E')
    ax = plt.gca()
    ax.set_facecolor('#1E1E1E')
    
    # Se configuran los estilos para los nodos centrales y vecinos
    center_nodes = [n for n, d in G.nodes(data=True) if d.get('is_center', False)]
    neigh_nodes = [n for n, d in G.nodes(data=True) if not d.get('is_center', False)]
    
    # Se dibuja el nodo central en color dorado y los vecindarios en azul cielo
    nx.draw_networkx_nodes(G, pos, nodelist=center_nodes, node_size=1100, node_color='#FFD700', edgecolors='#2D2D2D', linewidths=1.5, alpha=0.95)
    nx.draw_networkx_nodes(G, pos, nodelist=neigh_nodes, node_size=400, node_color='#3498DB', edgecolors='#2D2D2D', linewidths=1.2, alpha=0.85)
    
    # Se clasifican los enlaces en físicos e implícitos (semánticos)
    edges_phys = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') != 'semantic_similarity']
    edges_sem = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'semantic_similarity']
    
    # Se grafican los enlaces físicos como líneas grises continuas
    nx.draw_networkx_edges(G, pos, edgelist=edges_phys, width=1.5, edge_color='#5A5A5A', alpha=0.5)
    # Se grafican los enlaces semánticos como líneas discontinuas rojas
    nx.draw_networkx_edges(G, pos, edgelist=edges_sem, width=2.5, edge_color='#FF4C4C', alpha=0.85, style='dashed')
    
    # Se construyen las etiquetas enriquecidas de los nodos
    labels = {}
    for n in G.nodes:
        name = G.nodes[n].get('name', n)
        is_center = G.nodes[n].get('is_center', False)
        
        lbl = f"{name}\n({n})"
        if outlier_map.get(n, 0) == 1:
            lbl += "\n[ANÓMALO]"
            
        details = get_entity_evidence_details(n, df_evidence)
        if details:
            if details['lists']:
                lbl += f"\nListas: {','.join(details['lists'])}"
            if details['status']:
                lbl += f"\nEstado: {details['status']}"
            if details['reasons'] and is_center:
                r_text = "; ".join(details['reasons'])
                lbl += f"\nMotivo:\n" + textwrap.fill(r_text, width=30)
            elif details['reasons']:
                r_text = details['reasons'][0]
                if len(r_text) > 30:
                    r_text = r_text[:27] + "..."
                lbl += f"\nMotivo: {r_text}"
        elif not is_center and outlier_map.get(n, 0) == 0:
            lbl += "\n(Nodo Ordinario)"
            
        labels[n] = lbl
        
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color='#FFFFFF', font_family='sans-serif',
                           bbox=dict(facecolor='#1E1E1E', alpha=0.9, edgecolor='#555555', boxstyle='round,pad=0.2'))
                           
    # Se construyen las etiquetas detalladas para las aristas (patrón o vínculo identificado)
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        rel = d.get('relation')
        weight = d.get('weight', 1.0)
        
        if rel == 'semantic_similarity':
            edge_labels[(u, v)] = f"{weight:.1%}"
        else:
            shared_urls = set()
            shared_hashes = set()
            if not df_evidence.empty:
                shared_urls = set(df_evidence[df_evidence['entity_id'] == u]['url_or_reference'].dropna()) & set(df_evidence[df_evidence['entity_id'] == v]['url_or_reference'].dropna())
                shared_hashes = set(df_evidence[df_evidence['entity_id'] == u]['raw_content_hash'].dropna()) & set(df_evidence[df_evidence['entity_id'] == v]['raw_content_hash'].dropna())
            
            if shared_urls:
                url_val = list(shared_urls)[0]
                if "exports/SDN.XML" in url_val:
                    url_short = "OFAC SDN List"
                else:
                    url_short = url_val.split("//")[-1].split("/")[0]
                edge_labels[(u, v)] = f"Ref: {url_short}"
            elif shared_hashes:
                hash_val = list(shared_hashes)[0]
                edge_labels[(u, v)] = f"Hash: {hash_val[:6]}"
            else:
                edge_labels[(u, v)] = "Físico"
                
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, font_color='#FFD700',
                                bbox=dict(facecolor='#2D2D2D', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.1'))
    
    plt.title(f"Red Ego de Relaciones: {ent_name}\n({ent_id}) | Fecha: {RUN_DATE}", color='#FFFFFF', fontsize=11, fontweight='bold', pad=15)
    plt.axis('off')
    plt.xlim(-2.8, 2.8)
    plt.ylim(-2.8, 2.8)
    
    # Se genera la leyenda descriptiva del grafo
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
    Se orquesta la generación paralela de representaciones gráficas individuales de entidades utilizando multiprocesamiento.
    """
    logger.info("Generación de visualizaciones ego de red en progreso.")
    
    nodes_path = RUN_DIR / 'graph_enriched_entities.csv'
    if not nodes_path.exists():
        nodes_path = RUN_DIR / 'consolidated_entities.csv'
        
    edges_path = RUN_DIR / 'entity_edges.csv'
    
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError("No se encontraron los datos del grafo. Ejecute el pipeline primero.")
        
    df_nodes = pd.read_csv(nodes_path)
    df_edges = pd.read_csv(edges_path)
    
    # Se asegura la existencia del directorio destino de los gráficos
    output_dir = RUN_DIR / 'entity_graphs'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ruta de destino de gráficos: {output_dir}")
    
    name_map = dict(zip(df_nodes['entity_id'], df_nodes['entity_name']))
    
    # Se agrupan los enlaces por nodo para optimizar búsquedas relacionales
    from collections import defaultdict
    adj = defaultdict(list)
    for s, t, rel, w in zip(df_edges['source'].values, df_edges['target'].values, df_edges['relation_type'].values, df_edges['weight'].values):
        adj[s].append((t, rel, w, True))
        adj[t].append((s, rel, w, False))
        
    # Cargar test.csv y obtener las primeras 5 entidades
    test_csv_path = DATA_RAW_DIR / "test.csv"
    if test_csv_path.exists():
        df_test = pd.read_csv(test_csv_path)
        target_entity_ids = df_test['entity_id'].head(5).dropna().tolist()
        logger.info(f"Entidades objetivo de test.csv para visualización: {target_entity_ids}")
    else:
        target_entity_ids = []
        logger.warning(f"No se encontró test.csv en {test_csv_path}. Se usará comportamiento por defecto.")

    # Se recopilan las tareas a procesar en paralelo
    tasks = []
    
    # Si tenemos entidades objetivo, las filtramos y ordenamos según target_entity_ids
    if target_entity_ids:
        df_nodes_sorted = df_nodes[df_nodes['entity_id'].isin(target_entity_ids)].copy()
        df_nodes_sorted['entity_id'] = pd.Categorical(df_nodes_sorted['entity_id'], categories=target_entity_ids, ordered=True)
        df_nodes_sorted = df_nodes_sorted.sort_values('entity_id')
    else:
        # Se ordenan las entidades priorizando sospecha y PageRank si no hay target_entity_ids
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
            
    # Se limita el volumen a las 10 entidades más críticas para optimizar recursos si no hay filtro por test.csv
    if not target_entity_ids:
        max_graphs = 10
        if len(tasks) > max_graphs:
            logger.info(f"Se limita la visualización a las {max_graphs} entidades más críticas.")
            tasks = tasks[:max_graphs]
            
    logger.info(f"Total de grafos a generar: {len(tasks)}")
    
    # Se configura el pool de procesos de acuerdo a la capacidad del procesador
    import multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Multiprocessing Pool activo (cores: {num_workers}).")
    
    t0 = time.time()
    with multiprocessing.Pool(processes=num_workers) as pool:
        count = 0
        for _ in pool.imap_unordered(plot_single_entity_graph, tasks, chunksize=25):
            count += 1
            if count % 1000 == 0:
                logger.info(f"Progreso de generación: {count}/{len(tasks)}")
                
    elapsed = time.time() - t0
    logger.info(f"Visualización completada en {elapsed:.2f}s (promedio: {elapsed/len(tasks):.4f}s/img).")

if __name__ == '__main__':
    generate_all_entity_graphs()
