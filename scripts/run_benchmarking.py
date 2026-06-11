import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Añadir la raíz del proyecto al path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import logger, DATA_PROCESSED_DIR

def to_markdown_table(df: pd.DataFrame) -> str:
    """
    Convierte un DataFrame de Pandas a una cadena en formato de tabla Markdown.
    Evita la necesidad de dependencias externas como 'tabulate'.
    """
    headers = df.columns.tolist()
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    data_rows = []
    for _, row in df.iterrows():
        row_str = []
        for val in row.values:
            if isinstance(val, float):
                row_str.append(f"{val:.4f}")
            else:
                row_str.append(str(val))
        data_rows.append("| " + " | ".join(row_str) + " |")
        
    return "\n".join([header_row, separator_row] + data_rows)

def run_benchmarking_reporting():
    """
    Consolida las métricas de los módulos de agrupamiento semántico, detección de anomalías
    y análisis de grafos no supervisados en un reporte formal de descubrimiento de patrones.
    """
    logger.info("Iniciando consolidación de métricas de patrones no supervisados...")
    
    # Rutas de los archivos de métricas
    clustering_path = DATA_PROCESSED_DIR / 'clustering_metrics.csv'
    anomaly_path = DATA_PROCESSED_DIR / 'anomaly_metrics.csv'
    nodes_path = DATA_PROCESSED_DIR / 'graph_enriched_entities.csv'
    edges_path = DATA_PROCESSED_DIR / 'entity_edges.csv'
    comm_path = DATA_PROCESSED_DIR / 'graph_communities.csv'
    links_path = DATA_PROCESSED_DIR / 'hidden_entity_links.csv'
    
    # Cargar y verificar
    df_cluster = pd.read_csv(clustering_path) if clustering_path.exists() else None
    df_anom = pd.read_csv(anomaly_path) if anomaly_path.exists() else None
    df_nodes = pd.read_csv(nodes_path) if nodes_path.exists() else (pd.read_csv(DATA_PROCESSED_DIR / 'consolidated_entities.csv') if (DATA_PROCESSED_DIR / 'consolidated_entities.csv').exists() else None)
    df_edges = pd.read_csv(edges_path) if edges_path.exists() else None
    df_comm = pd.read_csv(comm_path) if comm_path.exists() else None
    df_links = pd.read_csv(links_path) if links_path.exists() else None
    
    report_content = f"""# REPORTE DE DESCUBRIMIENTO DE PATRONES Y VÍNCULOS AML OCULTOS
Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Enfoque: Análisis No Supervisado Puro (Sin Clasificación ni Puntuación de Riesgo CNBV)

Este reporte detalla las estructuras latentes, los vínculos ocultos identificados entre entidades mediante similitud semántica y las anomalías de información detectadas cuantitativamente.

---

## 1. MÓDULO 1: Agrupamiento Semántico de Perfiles de Entidades
Propósito: Evaluar sentence-transformers locales contra representaciones agregadas de perfiles de entidad.
"""

    if df_cluster is not None:
        report_content += "\n### Tabla Comparativa de Calidad de Clústeres (Representación Semántica)\n"
        report_content += to_markdown_table(df_cluster)
        report_content += "\n\n*Nota: HDBSCAN muestra un Silhouette superior y separa el ruido de forma efectiva, mientras que K-Means ofrece una segmentación completa y constante sin descartar datos.*\n"
    else:
        report_content += "\n[Métricas de Clustering no disponibles. Ejecute embeddings_clustering.py]\n"

    report_content += """
---

## 2. MÓDULO 2: Detección No Supervisada de Anomalías de Comportamiento e Información
Propósito: Identificar perfiles de comportamiento inusuales y outliers textuales o numéricos sin etiquetas de validación.
"""

    if df_anom is not None:
        report_content += "\n### Tabla de Anomalías Cuantitativas Detectadas\n"
        report_content += to_markdown_table(df_anom)
        report_content += "\n\n*Nota: Los modelos identidican un porcentaje fijo (5%) de entidades con desviaciones de comportamiento operacional en variables OSINT. Adicionalmente, el módulo de embeddings inyecta el flag de outlier de información en cada entidad.*\n"
    else:
        report_content += "\n[Métricas de Anomalías no disponibles. Ejecute anomaly_detection.py]\n"

    report_content += """
---

## 3. MÓDULO 3: Red Topológica y Vínculos Semánticos Ocultos (Análisis de Grafos)
Propósito: Mapear la conectividad implícita de las entidades y agruparlas en comunidades relacionales.
"""

    if df_nodes is not None and df_edges is not None:
        # Calcular estadísticas
        total_nodes = len(df_nodes)
        
        # Filtrar relaciones
        shared_ref = len(df_edges[df_edges['relation_type'] == 'shared_reference'])
        shared_cont = len(df_edges[df_edges['relation_type'] == 'shared_content'])
        sem_sim = len(df_edges[df_edges['relation_type'] == 'semantic_similarity'])
        total_edges = len(df_edges)
        num_communities = len(df_comm) if df_comm is not None else 0
        
        df_stats = pd.DataFrame([{
            'Métrica': 'Total de Entidades (Nodos)',
            'Valor': total_nodes
        }, {
            'Métrica': 'Relaciones por Referencias Compartidas',
            'Valor': shared_ref
        }, {
            'Métrica': 'Relaciones por Contenido Compartido',
            'Valor': shared_cont
        }, {
            'Métrica': 'Vínculos Semánticos Ocultos Detectados (Embeddings)',
            'Valor': sem_sim
        }, {
            'Métrica': 'Total de Enlaces en el Grafo',
            'Valor': total_edges
        }, {
            'Métrica': 'Comunidades Estructurales Identificadas (Louvain)',
            'Valor': num_communities
        }])
        
        report_content += "\n### Estadísticas Topológicas del Grafo de Entidades\n"
        report_content += to_markdown_table(df_stats)
        
        if df_comm is not None:
            report_content += "\n\n### Detalle de Comunidades de Red Louvain Detectadas (Top 5)\n"
            report_content += to_markdown_table(df_comm.head(5))
    else:
        report_content += "\n[Datos del Grafo no disponibles. Ejecute graph_analysis.py]\n"

    # Escribir el reporte en processed
    report_output_path = DATA_PROCESSED_DIR / 'benchmarking_report.md'
    with open(report_output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    logger.info(f"Reporte de Descubrimiento de Patrones exportado en: {report_output_path}")
    print("\n" + "="*80)
    import sys
    out_encoding = sys.stdout.encoding or 'utf-8'
    print(report_content.encode(out_encoding, errors='replace').decode(out_encoding))
    print("="*80 + "\n")

if __name__ == '__main__':
    run_benchmarking_reporting()
