import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Se añade la raíz del proyecto al path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import logger, RUN_DIR, RUN_DATE, PROJECT_ROOT, SHARED_DIR

def plot_clustering_comparison(df_cluster: pd.DataFrame, output_path: Path):
    """
    Se genera y guarda el gráfico comparativo de barras para los modelos de clustering.
    """
    try:
        import matplotlib
        try:
            import matplotlib.pyplot as plt
        except Exception:
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        
        df = df_cluster.copy().fillna(0)
        if df.empty:
            return
            
        model_names = df['Algoritmo'].tolist()
        silhouette_scores = df['Silhouette_Score'].tolist()
        cohesion_scores = df['Cohesion'].tolist()
        
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#1E1E1E')
        ax.set_facecolor('#1E1E1E')
        
        ax.bar(x - width/2, silhouette_scores, width, label='Silhouette Score', color='#2ECC71')
        ax.bar(x + width/2, cohesion_scores, width, label='Cohesión (Dist. Media)', color='#3498DB')
        
        ax.set_ylabel('Valor de Métrica', color='#FFFFFF')
        ax.set_title('Comparativa de Modelos de Clustering Semántico', color='#FFFFFF', fontsize=11, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right', color='#FFFFFF', fontsize=8)
        ax.tick_params(colors='#FFFFFF')
        ax.spines['bottom'].set_color('#555555')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#555555')
        ax.legend(facecolor='#2D2D2D', edgecolor='none', labelcolor='#FFFFFF', fontsize=8)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=120, facecolor='#1E1E1E')
        
        try:
            plt.show()
        except Exception:
            pass
            
        plt.close()
        logger.info(f"Gráfico de clustering guardado en: {output_path}")
    except Exception as e:
        logger.warning(f"No se pudo generar el gráfico de clustering: {e}")

def plot_anomaly_comparison(df_anom: pd.DataFrame, output_path: Path):
    """
    Se genera y guarda el gráfico comparativo de barras para los modelos de detección de anomalías.
    """
    try:
        import matplotlib
        try:
            import matplotlib.pyplot as plt
        except Exception:
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        
        df = df_anom.copy().fillna(0)
        if df.empty:
            return
            
        model_names = df['Algoritmo'].tolist()
        
        fig, ax = plt.subplots(figsize=(11, 5))
        fig.patch.set_facecolor('#1E1E1E')
        ax.set_facecolor('#1E1E1E')
        
        if 'PR-AUC' in df.columns and 'F1-Score' in df.columns and 'ROC-AUC' in df.columns:
            # Caso con etiquetas: Graficación de desempeño supervisado
            pr_aucs = df['PR-AUC'].tolist()
            f1_scores = df['F1-Score'].tolist()
            roc_aucs = df['ROC-AUC'].tolist()
            
            x = np.arange(len(model_names))
            width = 0.25
            
            ax.bar(x - width, pr_aucs, width, label='PR-AUC', color='#E74C3C')
            ax.bar(x, f1_scores, width, label='F1-Score', color='#E67E22')
            ax.bar(x + width, roc_aucs, width, label='ROC-AUC', color='#3498DB')
            
            ax.set_ylabel('Valor de Métrica', color='#FFFFFF')
            ax.set_title('Comparativa de Desempeño de Detección de Anomalías (Supervisada)', color='#FFFFFF', fontsize=11, fontweight='bold', pad=12)
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=15, ha='right', color='#FFFFFF', fontsize=8)
            ax.legend(facecolor='#2D2D2D', edgecolor='none', labelcolor='#FFFFFF', fontsize=8)
        elif 'Silhouette_Score' in df.columns and 'Distancia_Media_Outliers' in df.columns:
            # Caso no supervisado con métricas de separabilidad: Silhouette y Distancia Media
            sil_scores = df['Silhouette_Score'].tolist()
            dists = df['Distancia_Media_Outliers'].tolist()
            
            x = np.arange(len(model_names))
            width = 0.35
            
            ax.bar(x - width/2, sil_scores, width, label='Silhouette Score (Separabilidad)', color='#2ECC71')
            ax.bar(x + width/2, dists, width, label='Distancia Media Outliers', color='#3498DB')
            
            ax.set_ylabel('Valor de Métrica', color='#FFFFFF')
            ax.set_title('Comparativa de Modelos de Anomalías (Separabilidad y Distancia)', color='#FFFFFF', fontsize=11, fontweight='bold', pad=12)
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=15, ha='right', color='#FFFFFF', fontsize=8)
            ax.legend(facecolor='#2D2D2D', edgecolor='none', labelcolor='#FFFFFF', fontsize=8)
        else:
            # Caso no supervisado básico: volumen de anomalías
            anomalies_detected = df['Num_Anomalias_Detectadas'].tolist()
            x = np.arange(len(model_names))
            width = 0.5
            
            ax.bar(x, anomalies_detected, width, label='Anomalías Detectadas', color='#E74C3C')
            ax.set_ylabel('Cantidad de Anomalías', color='#FFFFFF')
            ax.set_title('Comparativa de Modelos de Detección de Anomalías (Volumen)', color='#FFFFFF', fontsize=11, fontweight='bold', pad=12)
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=15, ha='right', color='#FFFFFF', fontsize=8)
            ax.legend(facecolor='#2D2D2D', edgecolor='none', labelcolor='#FFFFFF', fontsize=8)
            
        ax.tick_params(colors='#FFFFFF')
        ax.spines['bottom'].set_color('#555555')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#555555')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=120, facecolor='#1E1E1E')
        
        try:
            plt.show()
        except Exception:
            pass
            
        plt.close()
        logger.info(f"Gráfico de anomalías guardado en: {output_path}")
    except Exception as e:
        logger.warning(f"No se pudo generar el gráfico de anomalías: {e}")

def to_markdown_table(df: pd.DataFrame) -> str:
    """
    Se convierte un DataFrame de Pandas a formato de tabla de Markdown.
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
    Se consolidan las métricas de agrupamiento semántico, detección de anomalías
    y análisis de grafos no supervisados en un reporte consolidado de descubrimiento de patrones.
    """
    run_dir_rel = RUN_DIR.relative_to(PROJECT_ROOT)
    
    # Se intenta generar las visualizaciones individuales por Entity ID
    try:
        from src.visualization import generate_all_entity_graphs
        generate_all_entity_graphs()
    except Exception as e:
        logger.error(f"Error al generar las visualizaciones de red por entidad: {e}")

    # Se intenta ejecutar el análisis relacional de cadenas
    try:
        from src.chain_analysis import run_chain_analysis
        run_chain_analysis()
    except Exception as e:
        logger.error(f"Error al ejecutar el análisis de cadenas: {e}")

    logger.info("Consolidación de métricas en progreso.")
    
    # Se definen las rutas de los archivos de métricas
    clustering_path = RUN_DIR / 'clustering_metrics.csv'
    anomaly_path = RUN_DIR / 'anomaly_metrics.csv'
    nodes_path = RUN_DIR / 'graph_enriched_entities.csv'
    edges_path = RUN_DIR / 'entity_edges.csv'
    comm_path = RUN_DIR / 'graph_communities.csv'
    links_path = RUN_DIR / 'hidden_entity_links.csv'
    chains_path = RUN_DIR / 'suspicious_chains.csv'
    loops_path = RUN_DIR / 'suspicious_loops.csv'
    
    # Se cargan y verifican los conjuntos de datos de métricas
    df_cluster = pd.read_csv(clustering_path) if clustering_path.exists() else None
    df_anom = pd.read_csv(anomaly_path) if anomaly_path.exists() else None
    df_nodes = pd.read_csv(nodes_path) if nodes_path.exists() else (pd.read_csv(RUN_DIR / 'consolidated_entities.csv') if (RUN_DIR / 'consolidated_entities.csv').exists() else None)
    df_edges = pd.read_csv(edges_path) if edges_path.exists() else None
    df_comm = pd.read_csv(comm_path) if comm_path.exists() else None
    df_links = pd.read_csv(links_path) if links_path.exists() else None
    df_chains = pd.read_csv(chains_path) if chains_path.exists() else None
    df_loops = pd.read_csv(loops_path) if loops_path.exists() else None
    
    # Se generan los gráficos comparativos no supervisados
    if df_cluster is not None:
        plot_clustering_comparison(df_cluster, RUN_DIR / 'clustering_model_comparison.png')
    if df_anom is not None:
        plot_anomaly_comparison(df_anom, RUN_DIR / 'anomaly_model_comparison.png')
        
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
        report_content += "\n\n![Comparativa de Modelos de Clustering](clustering_model_comparison.png)\n"
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
        if 'PR-AUC' in df_anom.columns:
            report_content += "\n\n*Nota: Se evalúa cuantitativamente el desempeño de cada detector de anomalías comparando sus salidas con las etiquetas del analista (`is_suspicious_analyst`). El gráfico de barras agrupadas presenta las métricas de PR-AUC, F1-Score y ROC-AUC para contrastar la efectividad de cada enfoque.*\n"
        elif 'Silhouette_Score' in df_anom.columns and 'Distancia_Media_Outliers' in df_anom.columns:
            report_content += "\n\n*Nota: Al no contar con etiquetas, se evalúan la Silhouette Score (indica qué tan bien separadas e aisladas quedan las anomalías del resto del grupo) y la Distancia Media (mide qué tan lejos están las anomalías del centroide normal). Valores más altos en ambas métricas indican que el modelo aísla de forma más clara las desviaciones extremas de comportamiento.*\n"
        else:
            report_content += "\n\n*Nota: Los modelos identifican un porcentaje fijo (5%) de entidades con desviaciones de comportamiento operacional en variables OSINT. El gráfico comparativo muestra el volumen total de anomalías identificadas por cada detector sobre el lote de datos analizado.*\n"
    else:
        report_content += "\n[Métricas de Anomalías no disponibles. Ejecute anomaly_detection.py]\n"

    report_content += """
---

## 2.5 MÓDULO 2.5: Aprendizaje Supervisado (Human-in-the-loop)
Propósito: Entrenar y comparar 9 clasificadores basados en decisiones históricas de analistas de cumplimiento para predecir la probabilidad de sospecha real y persistir el mejor modelo.
"""
    supervised_report = ""
    if df_nodes is not None and 'is_suspicious_analyst' in df_nodes.columns:
        logger.info("Entrenamiento y evaluación de los 9 modelos supervisados en progreso.")
        try:
            from sklearn.dummy import DummyClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
            from sklearn.svm import SVC
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score
            import joblib
            
            # Se preparan las variables explicativas y el vector objetivo
            features = ['evidence_items', 'max_identity_score', 'sources_with_hallazgo']
            features = [f for f in features if f in df_nodes.columns]
            
            X = df_nodes[features].fillna(0)
            y = df_nodes['is_suspicious_analyst'].fillna(0).astype(int)
            
            if len(np.unique(y)) > 1:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                
                # Definición de los 9 clasificadores a evaluar
                models = {
                    'baseline_0_dummy': DummyClassifier(strategy='stratified', random_state=42),
                    'baseline_1_logreg': LogisticRegression(max_iter=1000, random_state=42),
                    'baseline_2_tree': DecisionTreeClassifier(max_depth=5, random_state=42),
                    'baseline_3_rf': RandomForestClassifier(n_estimators=100, random_state=42),
                    'baseline_4_gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
                    'baseline_5_svm': SVC(probability=True, random_state=42),
                    'baseline_6_knn': KNeighborsClassifier(n_neighbors=5),
                    'ensemble_stacking': StackingClassifier(
                        estimators=[
                            ('rf', RandomForestClassifier(random_state=42)),
                            ('gb', GradientBoostingClassifier(random_state=42))
                        ],
                        final_estimator=LogisticRegression()
                    ),
                    'ensemble_voting': VotingClassifier(
                        estimators=[
                            ('rf', RandomForestClassifier(random_state=42)),
                            ('gb', GradientBoostingClassifier(random_state=42)),
                            ('lr', LogisticRegression(random_state=42))
                        ],
                        voting='soft'
                    )
                }
                
                comparison_data = []
                trained_instances = {}
                
                for name, clf in models.items():
                    t0 = time.time()
                    clf.fit(X_train, y_train)
                    
                    y_pred = clf.predict(X_test)
                    y_prob = clf.predict_proba(X_test)[:, 1]
                    elapsed_time = time.time() - t0
                    
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    roc_auc = roc_auc_score(y_test, y_prob)
                    
                    precision, recall, _ = precision_recall_curve(y_test, y_prob)
                    pr_auc = auc(recall, precision)
                    
                    comparison_data.append({
                        'Modelo': name,
                        'PR-AUC': pr_auc,
                        'F1-Score': f1,
                        'ROC-AUC': roc_auc,
                        'Tiempo_Segs': elapsed_time
                    })
                    trained_instances[name] = clf
                
                df_comparison = pd.DataFrame(comparison_data)
                
                # Se selecciona el mejor modelo según F1-Score
                best_row = df_comparison.sort_values(by='F1-Score', ascending=False).iloc[0]
                best_name = best_row['Modelo']
                best_model = trained_instances[best_name]
                
                # Se realiza la persistencia del mejor modelo ajustado
                joblib.dump(best_model, SHARED_DIR / 'models' / 'supervised_classifier.pkl')
                
                supervised_report = f"""
### MÓDULO 2.5: Benchmarking de Modelos Supervisados (Human-in-the-loop)
Se evalúan 9 clasificadores entrenados con etiquetas históricas del analista para determinar el modelo más confiable.

| Modelo | PR-AUC | F1-Score | ROC-AUC | Tiempo (s) |
| --- | --- | --- | --- | --- |
"""
                for _, row in df_comparison.iterrows():
                    supervised_report += f"| {row['Modelo']} | {row['PR-AUC']:.4f} | {row['F1-Score']:.4f} | {row['ROC-AUC']:.4f} | {row['Tiempo_Segs']:.4f} |\n"
                
                # Se incrusta el gráfico comparativo en el reporte Markdown
                supervised_report += "\n\n![Comparativa de Modelos Supervisados](supervised_model_comparison.png)\n"
                supervised_report += f"\n*Nota: El mejor modelo seleccionado por F1-Score es **{best_name}** y ha sido serializado en `shared/models/supervised_classifier.pkl`.*\n"
                
                # Imprimir la tabla en la consola estándar
                print("\n" + "="*80)
                print("TABLA COMPARATIVA DE DESEMPEÑO DE MODELOS SUPERVISADOS (CMS)")
                print(to_markdown_table(df_comparison))
                print(f"Modelo seleccionado como óptimo: {best_name}")
                print("="*80 + "\n")
                
                # Graficación de barras agrupadas
                try:
                    import matplotlib.pyplot as plt
                    
                    model_names = df_comparison['Modelo'].tolist()
                    pr_aucs = df_comparison['PR-AUC'].tolist()
                    f1_scores = df_comparison['F1-Score'].tolist()
                    roc_aucs = df_comparison['ROC-AUC'].tolist()
                    
                    x = np.arange(len(model_names))
                    width = 0.25
                    
                    fig, ax = plt.subplots(figsize=(11, 5))
                    fig.patch.set_facecolor('#1E1E1E')
                    ax.set_facecolor('#1E1E1E')
                    
                    rects1 = ax.bar(x - width, pr_aucs, width, label='PR-AUC', color='#E74C3C')
                    rects2 = ax.bar(x, f1_scores, width, label='F1-Score', color='#E67E22')
                    rects3 = ax.bar(x + width, roc_aucs, width, label='ROC-AUC', color='#3498DB')
                    
                    ax.set_ylabel('Valor de Métrica', color='#FFFFFF')
                    ax.set_title('Comparativa de Desempeño de Modelos Supervisados', color='#FFFFFF', fontsize=11, fontweight='bold', pad=12)
                    ax.set_xticks(x)
                    ax.set_xticklabels(model_names, rotation=20, ha='right', color='#FFFFFF', fontsize=8)
                    ax.tick_params(colors='#FFFFFF')
                    ax.spines['bottom'].set_color('#555555')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#555555')
                    ax.legend(facecolor='#2D2D2D', edgecolor='none', labelcolor='#FFFFFF', fontsize=8)
                    plt.tight_layout()
                    
                    # Se guarda el gráfico comparativo en el directorio de la corrida
                    plot_output_path = RUN_DIR / 'supervised_model_comparison.png'
                    plt.savefig(plot_output_path, dpi=120, facecolor='#1E1E1E')
                    logger.info(f"Gráfico de comparación supervisado guardado en: {plot_output_path}")
                    
                    try:
                        plt.show()
                    except Exception:
                        pass
                    plt.close()
                except Exception as ex_plot:
                    logger.warning(f"No se pudo desplegar la gráfica comparativa: {ex_plot}")
                    
            else:
                supervised_report = "\n[Datos insuficientes para entrenamiento supervisado (solo una clase presente en is_suspicious_analyst)]\n"
        except Exception as e:
            logger.error(f"Error entrenando clasificador supervisado: {e}")
            supervised_report = f"\n[Error entrenando clasificador supervisado: {e}]\n"
    else:
        supervised_report = """
### Estado del Modelo Supervisado: **DESACTIVADO**
*No se encontraron etiquetas de decisiones del analista ('is_suspicious_analyst') en los datos de entrada crudos.*

#### Requisitos de Habilitación para el Clasificador Supervisado en Producción:
1. **Extracción de Decisiones**: Se requiere exportar las resoluciones del analista desde el sistema de gestión de casos (CMS), asignando una etiqueta binaria:
   - `1`: Confirmado Sospechoso / Reporte de Operación Inusual (ROI) enviado.
   - `0`: Alerta Cerrada / Homónimo / Falso Positivo.
2. **Ingesta de Datos**: Se debe integrar esta columna bajo el nombre `is_suspicious_analyst` en el archivo de entrada `entity_match_summary.csv` en `data/raw/`.
3. **Entrenamiento Automatizado**: Al ejecutar los scripts `train_pipeline.py` o `use_pipeline.py`, el pipeline identificará la presencia de la columna, ajustará los modelos y registrará el mejor en `data/processed/shared/models/supervised_classifier.pkl` para su posterior uso predictivo.
"""
    report_content += supervised_report

    report_content += """
---

## 3. MÓDULO 3: Red Topológica y Vínculos Semánticos Ocultos (Análisis de Grafos)
Propósito: Mapear la conectividad implícita de las entidades y agruparlas en comunidades relacionales.
"""

    if df_nodes is not None and df_edges is not None:
        # Se calculan las estadísticas estructurales del grafo
        total_nodes = len(df_nodes)
        
        # Se filtran los tipos de relación para el análisis descriptivo
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

    # 4. Sección de Visualización
    report_content += f"""
---

## 4. ANÁLISIS VISUAL DE LA RED Y VÍNCULOS OCULTOS
Se han generado visualizaciones de red individuales (grafo ego de 1 salto) por cada Entity ID con conexiones en el siguiente directorio:

- **Directorio de Gráficos de Entidades**: [{run_dir_rel.as_posix()}/entity_graphs/](file:///{str((RUN_DIR / 'entity_graphs').as_posix())}/)

Cada imagen dispone de forma concéntrica a los vecinos alrededor de la entidad consultada (nodo dorado central), permitiendo identificar claramente:
- **Líneas grises continuas**: Conexiones físicas conocidas (URLs o hashes de contenido compartidos).
- **Líneas rojas discontinuas**: Vínculos de similitud semántica ocultos descubiertos por embeddings.
"""

    # 5. Sección de Análisis de Cadenas
    report_content += f"""
---

## 5. DETECCIÓN DE VÍNCULOS EN CADENA (MULTI-HOP) Y BUCLES SOSPECHOSOS
Propósito: Rastrear caminos de relación indirectos (hasta 3 saltos) desde listas de sanciones hacia entidades ordinarias/anómalas, y detectar ciclos de simulación.
"""

    if df_chains is not None and not df_chains.empty:
        report_content += "\n### Top 5 Caminos en Cadena Detectados a Entidades de Interés/Anómalas\n"
        df_show_chains = df_chains[['source_blacklist_name', 'target_entity_name', 'path_hops', 'path_nodes_names', 'avg_edge_weight']].head(5)
        df_show_chains.columns = ['Origen (Lista Negra)', 'Destino Ordinario', 'Saltos', 'Copia del Camino', 'Peso Promedio']
        report_content += to_markdown_table(df_show_chains)
        
        report_content += "\n\n### Diagramas de Cadenas Críticas Generadas:\n"
        for idx in range(min(5, len(df_chains))):
            chain_img_path = RUN_DIR / 'critical_chains' / f"critical_chain_{idx+1}_{RUN_DATE}.png"
            if chain_img_path.exists():
                report_content += f"- **Cadena #{idx+1}**: [{df_chains.iloc[idx]['source_blacklist_name']} -> {df_chains.iloc[idx]['target_entity_name']}](file:///{chain_img_path.as_posix()})\n"
    else:
        report_content += "\n[No se detectaron caminos de conexión en cadena significativos en este conjunto de datos]\n"

    if df_loops is not None and not df_loops.empty:
        report_content += "\n### Bucles Relacionales Cerrados Detectados (Triangulaciones)\n"
        df_show_loops = df_loops[['cycle_length', 'blacklist_count', 'outliers_count', 'cycle_node_ids', 'cycle_node_names']].head(5)
        df_show_loops.columns = ['Largo del Bucle', 'Semillas en Lista Negra', 'Nodos Anómalos', 'ID de Nodos', 'Nombre de Nodos']
        report_content += to_markdown_table(df_show_loops)
        report_content += "\n\n*Nota: Los bucles cerrados representan agrupaciones de entidades altamente vinculadas entre sí, útiles para identificar redes de empresas fantasma u operaciones simuladas.*\n"
    else:
        report_content += "\n[No se detectaron bucles relacionales cerrados en este conjunto de datos]\n"

    # Se escribe el reporte final en el directorio de la corrida
    report_output_path = RUN_DIR / 'benchmarking_report.md'
    with open(report_output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    logger.info(f"Reporte exportado en: {report_output_path}")
    print("\n" + "="*80)
    import sys
    out_encoding = sys.stdout.encoding or 'utf-8'
    print(report_content.encode(out_encoding, errors='replace').decode(out_encoding))
    print("="*80 + "\n")

if __name__ == '__main__':
    import argparse
    import os
    import importlib
    
    parser = argparse.ArgumentParser(description="Reporte de Descubrimiento de Patrones AML")
    parser.add_argument('--mode', type=str, choices=['train', 'use'], default=None,
                        help="Modo de ejecución: 'train' para entrenamiento, 'use' para uso/inferencia.")
    args, unknown = parser.parse_known_args()
    
    if args.mode:
        os.environ["PIPELINE_MODE"] = args.mode
        import src.config
        importlib.reload(src.config)
        # Se actualizan las variables de ejecución importadas
        from src.config import RUN_DIR, RUN_DATE, PROJECT_ROOT
        
    run_benchmarking_reporting()
