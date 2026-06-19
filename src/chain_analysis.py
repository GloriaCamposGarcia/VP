import os
import sys
import time
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from src.config import logger, RUN_DIR, RUN_DATE, DATA_RAW_DIR
import textwrap

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

def is_blacklist_node(node_id: str) -> bool:
    """
    Se identifica si un nodo pertenece a una lista de control de sanciones según su prefijo de origen.
    """
    blacklist_prefixes = [
        'OFAC_SDN', 'EU_FINANCIAL_SANCTIONS', 'UN_CONSOLIDATED', 
        'UK_SANCTIONS', 'WORLD_BANK_DEBARRED', 'IDB_SANCTIONED', 
        'SAT_69', 'SAT_69B'
    ]
    return any(str(node_id).startswith(pfx) for pfx in blacklist_prefixes)

def run_chain_analysis():
    """
    Se analizan los caminos relacionales indirectos (hasta 3 saltos) y bucles relacionales
    entre entidades sancionadas y perfiles operativos ordinarios o anómalos.
    """
    logger.info("Análisis de cadenas relacionales y caminos multi-salto en progreso.")
    
    nodes_path = RUN_DIR / 'graph_enriched_entities.csv'
    edges_path = RUN_DIR / 'entity_edges.csv'
    
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError("Debe ejecutar los scripts previos del pipeline para generar los archivos consolidados.")
        
    df_nodes = pd.read_csv(nodes_path)
    df_edges = pd.read_csv(edges_path)
    
    # Se cargan las columnas de anomalía desde consolidated_entities.csv si no están en graph_enriched_entities.csv
    consolidated_path = RUN_DIR / 'consolidated_entities.csv'
    if consolidated_path.exists():
        df_consolidated = pd.read_csv(consolidated_path)
        anomaly_cols = ['entity_id', 'anomaly_isolationforest', 'anomaly_oneclasssvm', 'anomaly_localoutlierfactor', 'is_embedding_outlier']
        existing_anomaly_cols = [c for c in anomaly_cols if c in df_consolidated.columns]
        if len(existing_anomaly_cols) > 1:
            cols_to_drop = [c for c in existing_anomaly_cols if c in df_nodes.columns and c != 'entity_id']
            if cols_to_drop:
                df_nodes = df_nodes.drop(columns=cols_to_drop)
            df_nodes = df_nodes.merge(df_consolidated[existing_anomaly_cols], on='entity_id', how='left')
    
    # Se crea el mapeo asociativo de identificador a nombre y tipo
    name_map = dict(zip(df_nodes['entity_id'], df_nodes['entity_name']))
    type_map = dict(zip(df_nodes['entity_id'], df_nodes['entity_type']))
    
    # Se determinan los outliers según el conjunto de modelos
    df_nodes['is_outlier'] = (
        (df_nodes.get('anomaly_isolationforest', 0) == 1) |
        (df_nodes.get('anomaly_oneclasssvm', 0) == 1) |
        (df_nodes.get('anomaly_localoutlierfactor', 0) == 1) |
        (df_nodes.get('is_embedding_outlier', 0) == 1)
    ).astype(int)
    
    outlier_map = dict(zip(df_nodes['entity_id'], df_nodes['is_outlier']))
    
    # 1. Se construye el grafo filtrado
    # Se filtra el ruido relacional proveniente de descargas globales de listas para evitar mega-cliques
    logger.info("Filtrado de ruido relacional global en progreso.")
    
    filtered_edges = df_edges[
        (df_edges['relation_type'] != 'shared_reference') | 
        (df_edges['weight'] >= 2.0)
    ]
    logger.info(f"Aristas filtradas: {len(filtered_edges)} (original: {len(df_edges)}).")
    
    G = nx.Graph()
    for _, row in df_nodes.iterrows():
        G.add_node(
            row['entity_id'], 
            name=row['entity_name'], 
            is_blacklist=is_blacklist_node(row['entity_id']),
            is_outlier=bool(outlier_map.get(row['entity_id'], 0))
        )
        
    for _, row in filtered_edges.iterrows():
        G.add_edge(
            row['source'], 
            row['target'], 
            relation=row['relation_type'], 
            weight=float(row['weight'])
        )
        
    # Se clasifican los conjuntos de nodos del grafo
    blacklist_nodes = [n for n in G.nodes if G.nodes[n].get('is_blacklist', False)]
    ordinary_nodes = [n for n in G.nodes if not G.nodes[n].get('is_blacklist', False)]
    anomalous_ordinary_nodes = [n for n in ordinary_nodes if G.nodes[n].get('is_outlier', False)]
    
    logger.info(f"Nodos en listas de control: {len(blacklist_nodes)}")
    logger.info(f"Nodos ordinarios: {len(ordinary_nodes)} (anómalos: {len(anomalous_ordinary_nodes)}).")
    
    # 2. Se realiza la búsqueda de caminos multi-salto (cadenas)
    chains = []
    logger.info("Búsqueda de caminos entre listas de control y anomalías en progreso.")
    
    # Se utiliza una búsqueda BFS desde los nodos semilla de listas de control
    path_count = 0
    t0 = time.time()
    
    for start_node in ordinary_nodes:
        if start_node not in G:
            continue
        # Se calculan los caminos más cortos desde el nodo ordinario/anómalo con un límite de profundidad de 3 saltos
        try:
            paths = nx.single_source_shortest_path(G, start_node, cutoff=3)
            for target_node, path in paths.items():
                # El destino debe ser un nodo de lista de control (blacklist)
                if len(path) >= 3 and G.nodes[target_node].get('is_blacklist', False):
                    is_start_outlier = G.nodes[start_node].get('is_outlier', False)
                    path_len = len(path) - 1
                    
                    # Se calcula el peso de enlace promedio de la trayectoria
                    weights = []
                    relations = []
                    for i in range(path_len):
                        u, v = path[i], path[i+1]
                        edge_data = G.edges[u, v]
                        weights.append(edge_data.get('weight', 1.0))
                        relations.append(edge_data.get('relation', 'unknown'))
                        
                    avg_weight = float(np.mean(weights))
                    
                    # Se evalúa el criterio de inclusión en el reporte de cadenas
                    if is_start_outlier or (path_len <= 2 and avg_weight >= 1.0):
                        # Se invierte el camino para reportar desde la lista de control hacia el nodo ordinario
                        reversed_path = list(reversed(path))
                        reversed_relations = list(reversed(relations))
                        
                        chains.append({
                            'source_blacklist_id': target_node,
                            'source_blacklist_name': name_map.get(target_node, target_node),
                            'target_entity_id': start_node,
                            'target_entity_name': name_map.get(start_node, start_node),
                            'is_target_anomalous': int(is_start_outlier),
                            'path_hops': path_len,
                            'path_nodes_ids': " -> ".join(reversed_path),
                            'path_nodes_names': " -> ".join([name_map.get(n, n) for n in reversed_path]),
                            'path_relations': " -> ".join(reversed_relations),
                            'avg_edge_weight': avg_weight
                        })
                        path_count += 1
        except Exception as e:
            logger.error(f"Error procesando caminos para {start_node}: {e}")
            
    df_chains = pd.DataFrame(chains)
    if not df_chains.empty:
        # Se ordenan los resultados priorizando anomalía y menor cantidad de saltos
        df_chains = df_chains.sort_values(
            by=['is_target_anomalous', 'path_hops', 'avg_edge_weight'], 
            ascending=[False, True, False]
        ).reset_index(drop=True)
        
        # Cargar test.csv y obtener las primeras 5 entidades
        test_csv_path = DATA_RAW_DIR / "test.csv"
        if test_csv_path.exists():
            df_test = pd.read_csv(test_csv_path)
            target_entity_ids = df_test['entity_id'].head(5).dropna().tolist()
            if target_entity_ids:
                # Filtrar cadenas cuyas fuentes estén en las 5 primeras entidades de test.csv
                df_chains = df_chains[df_chains['source_blacklist_id'].isin(target_entity_ids)].reset_index(drop=True)
                # Mantener solo una cadena (la más crítica) por cada entidad (drop duplicates)
                df_chains = df_chains.drop_duplicates(subset=['source_blacklist_id'], keep='first').reset_index(drop=True)
                logger.info(f"Cadenas críticas filtradas para test.csv: {len(df_chains)}")
        
        chains_path = RUN_DIR / 'suspicious_chains.csv'
        df_chains.to_csv(chains_path, index=False)
        logger.info(f"Caminos detectados: {len(df_chains)}. Reporte: {chains_path}")
    else:
        logger.info("Sin caminos indirectos detectados.")
        df_chains = pd.DataFrame(columns=[
            'source_blacklist_id', 'source_blacklist_name', 'target_entity_id', 'target_entity_name',
            'is_target_anomalous', 'path_hops', 'path_nodes_ids', 'path_nodes_names', 'path_relations', 'avg_edge_weight'
        ])
        df_chains.to_csv(RUN_DIR / 'suspicious_chains.csv', index=False)
 
    # 3. Se realiza la detección de ciclos o bucles relacionales
    logger.info("Búsqueda de ciclos relacionales en progreso.")
    cycles = []
    try:
        # Se calcula la base de ciclos del grafo para longitudes entre 3 y 5
        all_cycles = nx.cycle_basis(G)
        for cycle in all_cycles:
            if 3 <= len(cycle) <= 5:
                # Se cuantifica la presencia de nodos de listas de control y outliers
                blacklist_in_cycle = sum(1 for n in cycle if G.nodes[n].get('is_blacklist', False))
                outliers_in_cycle = sum(1 for n in cycle if G.nodes[n].get('is_outlier', False))
                
                cycle_names = [name_map.get(n, n) for n in cycle]
                cycles.append({
                    'cycle_length': len(cycle),
                    'blacklist_count': blacklist_in_cycle,
                    'outliers_count': outliers_in_cycle,
                    'cycle_node_ids': " - ".join(cycle) + " - " + cycle[0],
                    'cycle_node_names': " - ".join(cycle_names) + " - " + cycle_names[0]
                })
    except Exception as e:
        logger.error(f"Error en detección de ciclos: {e}")
        
    df_cycles = pd.DataFrame(cycles)
    if not df_cycles.empty:
        df_cycles = df_cycles.sort_values(
            by=['blacklist_count', 'outliers_count', 'cycle_length'], 
            ascending=[False, False, True]
        ).reset_index(drop=True)
        cycles_path = RUN_DIR / 'suspicious_loops.csv'
        df_cycles.to_csv(cycles_path, index=False)
        logger.info(f"Ciclos detectados: {len(df_cycles)}. Reporte: {cycles_path}")
    else:
        pd.DataFrame(columns=['cycle_length', 'blacklist_count', 'outliers_count', 'cycle_node_ids', 'cycle_node_names']).to_csv(
            RUN_DIR / 'suspicious_loops.csv', index=False
        )
 
    # 4. Se generan las representaciones gráficas para las cadenas críticas
    if not df_chains.empty:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        output_dir = RUN_DIR / 'critical_chains'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar df_evidence
        evidence_path = DATA_RAW_DIR / "evidence_items.csv"
        if evidence_path.exists():
            df_evidence = pd.read_csv(evidence_path)
        else:
            df_evidence = pd.DataFrame()
            
        top_chains = df_chains.head(5)
        logger.info("Generación de diagramas de cadenas críticas en progreso.")
        
        for idx, row in top_chains.iterrows():
            path_nodes = row['path_nodes_ids'].split(" -> ")
            relations = row['path_relations'].split(" -> ")
            
            # Se construye un subgrafo lineal dirigido para la representación horizontal
            path_G = nx.DiGraph()
            
            # Se agregan los nodos con sus tipos y atributos de posición
            for i, node_id in enumerate(path_nodes):
                is_bl = G.nodes[node_id].get('is_blacklist', False)
                is_out = G.nodes[node_id].get('is_outlier', False)
                path_G.add_node(
                    node_id, 
                    name=name_map.get(node_id, node_id),
                    node_type="blacklist" if is_bl else ("outlier" if is_out else "ordinary"),
                    step=i
                )
                
            # Se agregan las aristas de relación
            for i in range(len(path_nodes) - 1):
                path_G.add_edge(
                    path_nodes[i], 
                    path_nodes[i+1], 
                    relation=relations[i]
                )
                
            # Se define el layout horizontal con espaciado constante más ancho para evitar solapamientos
            pos = {}
            for node in path_G.nodes:
                step = path_G.nodes[node]['step']
                pos[node] = np.array([float(step) * 3.5, 0.0])
                
            # Se genera el lienzo de visualización más amplio para alojar texto descriptivo
            fig = plt.figure(figsize=(14.0, 5.0), facecolor='#1E1E1E')
            ax = plt.gca()
            ax.set_facecolor('#1E1E1E')
            
            # Se asignan los colores de los nodos según su tipo
            node_colors = []
            for node in path_G.nodes:
                nt = path_G.nodes[node]['node_type']
                if nt == "blacklist":
                    node_colors.append('#E74C3C')
                elif nt == "outlier":
                    node_colors.append('#E67E22')
                else:
                    node_colors.append('#3498DB')
                    
            # Se dibujan los nodos del camino
            nx.draw_networkx_nodes(path_G, pos, node_size=1200, node_color=node_colors, edgecolors='#2D2D2D', linewidths=1.5, alpha=0.95)
            
            # Se dibujan las aristas dirigidas del camino con etiquetas descriptivas
            for u, v, d in path_G.edges(data=True):
                rel = d.get('relation', 'unknown')
                weight = G.edges[u, v].get('weight', 1.0)
                color = '#FF4C4C' if rel == 'semantic_similarity' else '#95A5A6'
                style = 'dashed' if rel == 'semantic_similarity' else 'solid'
                
                nx.draw_networkx_edges(
                    path_G, pos, edgelist=[(u, v)], width=2.0, 
                    edge_color=color, style=style, arrows=True, 
                    arrowsize=15, connectionstyle="arc3,rad=0.0"
                )
                
                # Se busca si comparten URL o Hash en caso de enlace físico
                shared_urls = set()
                shared_hashes = set()
                if not df_evidence.empty:
                    shared_urls = set(df_evidence[df_evidence['entity_id'] == u]['url_or_reference'].dropna()) & set(df_evidence[df_evidence['entity_id'] == v]['url_or_reference'].dropna())
                    shared_hashes = set(df_evidence[df_evidence['entity_id'] == u]['raw_content_hash'].dropna()) & set(df_evidence[df_evidence['entity_id'] == v]['raw_content_hash'].dropna())
                
                if rel == 'semantic_similarity':
                    rel_label = f"Vínculo Semántico\n(Simil: {weight:.1%})"
                elif shared_urls:
                    url_val = list(shared_urls)[0]
                    if "exports/SDN.XML" in url_val:
                        url_short = "OFAC SDN List"
                    else:
                        url_short = url_val.split("//")[-1].split("/")[0]
                    rel_label = f"Ref. Compartida\n({url_short})"
                elif shared_hashes:
                    hash_val = list(shared_hashes)[0]
                    rel_label = f"Contenido Compartido\n(Hash: {hash_val[:6]})"
                else:
                    rel_label = f"Enlace Físico ({rel})"
                    
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = 0.08
                plt.text(
                    mid_x, mid_y, rel_label, color='#FFFFFF', fontsize=7, 
                    ha='center', va='bottom', fontweight='semibold',
                    bbox=dict(facecolor='#2D2D2D', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2')
                )
                
            # Se dibujan las etiquetas identificadoras y de regulación de las entidades
            labels = {}
            for n in path_G.nodes:
                name = path_G.nodes[n]['name']
                is_bl = path_G.nodes[n]['node_type'] == 'blacklist'
                is_out = path_G.nodes[n]['node_type'] == 'outlier'
                
                lbl = f"{name}\n({n})"
                if is_out:
                    lbl += "\n[ANÓMALO]"
                    
                details = get_entity_evidence_details(n, df_evidence)
                if details:
                    if details['lists']:
                        lbl += f"\nListas: {','.join(details['lists'])}"
                    if details['status']:
                        lbl += f"\nEstado: {details['status']}"
                    if details['reasons']:
                        r_text = "; ".join(details['reasons'])
                        lbl += f"\nMotivo:\n" + textwrap.fill(r_text, width=32)
                elif not is_bl and not is_out:
                    lbl += "\n(Nodo Ordinario)"
                    
                labels[n] = lbl
                
            nx.draw_networkx_labels(
                path_G, pos, labels=labels, font_size=7, font_color='#FFFFFF', 
                font_family='sans-serif',
                verticalalignment='top',
                bbox=dict(facecolor='#1E1E1E', alpha=0.95, edgecolor='#555555', boxstyle='round,pad=0.3')
            )
            
            plt.title(f"Cadena Crítica #{idx+1} (Saltos: {row['path_hops']})\nOrigen: Lista Negra de Control | Fecha: {RUN_DATE}", color='#FFFFFF', fontsize=11, fontweight='bold', pad=15)
            plt.axis('off')
            plt.xlim(-1.2, float(len(path_nodes) - 1) * 3.5 + 1.2)
            plt.ylim(-1.2, 1.2)
            
            out_img_path = output_dir / f"critical_chain_{idx+1}_{RUN_DATE}.png"
            plt.savefig(out_img_path, dpi=120, facecolor='#1E1E1E', bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Diagrama de cadena guardado en: {out_img_path}")
            
    logger.info("Análisis de cadenas y ciclos finalizado.")

if __name__ == '__main__':
    run_chain_analysis()
