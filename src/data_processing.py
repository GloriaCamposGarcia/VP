import ast
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from src.config import logger, DATA_RAW_DIR, DATA_PROCESSED_DIR, RUN_DIR, PIPELINE_MODE

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Se realiza la carga de los conjuntos de datos correspondientes a los resultados de fuentes,
    evidencias y el resumen de coincidencias desde la ruta de datos de origen (raw).
    Se contempla el soporte para directorios específicos según el modo de ejecución (train/use).
    """
    raw_dir = DATA_RAW_DIR / PIPELINE_MODE if (DATA_RAW_DIR / PIPELINE_MODE).exists() else DATA_RAW_DIR
    logger.info(f"Carga de datos crudos desde: {raw_dir}")

    source_results_path = raw_dir / 'entity_source_results.csv'
    evidence_items_path = raw_dir / 'evidence_items.csv'
    match_summary_path = raw_dir / 'entity_match_summary.csv'

    if not source_results_path.exists() or not evidence_items_path.exists() or not match_summary_path.exists():
        raise FileNotFoundError(
            f"Falta alguno de los conjuntos de datos requeridos en {raw_dir}."
        )

    logger.info("Carga de conjuntos de datos AML/OSINT en progreso.")
    df_sources = pd.read_csv(source_results_path)
    df_evidence = pd.read_csv(evidence_items_path)
    df_match_summary = pd.read_csv(match_summary_path)
    logger.info(
        f"Resultados de fuentes: {df_sources.shape}, "
        f"Evidencias: {df_evidence.shape}, "
        f"Resumen: {df_match_summary.shape}"
    )
    return df_sources, df_evidence, df_match_summary

def parse_entity_catalog(
    df_sources: pd.DataFrame, 
    df_evidence: pd.DataFrame = None, 
    df_match_summary: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Se extrae y unifica un catálogo de entidades únicas a partir de las fuentes disponibles.
    Se prioriza la información del resumen de coincidencias. Como alternativa de respaldo,
    se analizan la columna 'query_used' de los resultados de fuentes y el catálogo de evidencias.
    """
    logger.info("Construcción del catálogo de entidades en progreso.")
    entities: Dict[str, dict] = {}
    
    # Se construye un mapa dinámico de país a código a partir de df_match_summary
    dynamic_country_mapping = {}
    if df_match_summary is not None and not df_match_summary.empty:
        for _, row in df_match_summary.iterrows():
            code = str(row.get('input_country', '')).strip().upper()
            matches_str = row.get('matches_json', '')
            if pd.notna(code) and code not in ['UNKNOWN', ''] and pd.notna(matches_str) and matches_str.strip():
                try:
                    import json
                    matches = json.loads(matches_str)
                    for m in matches:
                        c_name = m.get('country')
                        if c_name:
                            dynamic_country_mapping[str(c_name).strip().lower()] = code
                except Exception:
                    pass

    def map_country(c_str: str) -> str:
        c_clean = str(c_str).strip()
        if not c_clean or c_clean.lower() in ['unknown', '', 'nan']:
            return 'DESCONOCIDO'
        # Se verifica si está en el mapeo dinámico construido desde la data cruda
        if c_clean.lower() in dynamic_country_mapping:
            return dynamic_country_mapping[c_clean.lower()]
        # Se mantiene si ya es un código de 3 letras
        if len(c_clean) == 3 and c_clean.isalpha():
            return c_clean.upper()
        # Fallback dinámico: primeras 3 letras en mayúsculas
        return c_clean[:3].upper()

    # 1. Se prioriza la información de df_match_summary
    if df_match_summary is not None and not df_match_summary.empty:
        logger.info("Carga de metadatos desde resumen de coincidencias.")
        for _, row in df_match_summary.iterrows():
            ent_id = row['entity_id']
            name = row['entity_name']
            raw_type = row['input_entity_type']
            country = row['input_country']
            
            # Se normaliza el tipo de entidad según la práctica CNBV (FISICA o MORAL)
            ent_type = 'MORAL' if raw_type == 'PM' else ('FISICA' if raw_type == 'PF' else 'OTRO')
            country_code = map_country(country)
                
            if ent_id and name and ent_id not in entities:
                entities[ent_id] = {
                    'entity_id': ent_id,
                    'entity_name': name,
                    'entity_type': ent_type,
                    'country_code': country_code
                }

    # 2. Se complementa con df_sources ('query_used') para entidades sin coincidencias
    logger.info("Búsqueda de entidades adicionales en query_used.")
    for val in df_sources['query_used'].dropna().unique():
        try:
            data = ast.literal_eval(val)
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                ent_id = item.get('entity_id')
                name = item.get('query_value')
                metadata = item.get('metadata', {})
                country = metadata.get('country_code', '')
                raw_type = metadata.get('entity_type', '')
                
                if ent_id and ent_id not in entities:
                    ent_type = 'MORAL' if raw_type == 'PM' else ('FISICA' if raw_type == 'PF' else 'OTRO')
                    entities[ent_id] = {
                        'entity_id': ent_id,
                        'entity_name': name,
                        'entity_type': ent_type,
                        'country_code': map_country(country) if country else 'DESCONOCIDO'
                    }
        except Exception as e:
            continue
            
    df_entities = pd.DataFrame(list(entities.values()))
    
    # 3. Se resuelven casos huérfanos con df_evidence por seguridad
    unique_ids_sources = df_sources['entity_id'].dropna().unique()
    missing_ids = set(unique_ids_sources) - set(df_entities['entity_id']) if not df_entities.empty else set(unique_ids_sources)
    if missing_ids:
        logger.info(f"Resolución de país para {len(missing_ids)} entidades sin metadatos.")
        missing_rows = []
        
        for mid in missing_ids:
            detected_country = 'DESCONOCIDO'
            
            if df_evidence is not None and not df_evidence.empty and 'country' in df_evidence.columns:
                sub_ev = df_evidence[df_evidence['entity_id'] == mid]
                if not sub_ev.empty:
                    raw_val = str(sub_ev['country'].iloc[0]).strip()
                    if raw_val:
                        detected_country = map_country(raw_val)
            
            missing_rows.append({
                'entity_id': mid,
                'entity_name': f"Entidad Desconocida {mid}",
                'entity_type': 'OTRO',
                'country_code': detected_country
            })
        df_entities = pd.concat([df_entities, pd.DataFrame(missing_rows)], ignore_index=True)

    logger.info(f"Catálogo construido. Total entidades únicas: {len(df_entities)}")
    return df_entities

def consolidate_entity_features(
    df_entities: pd.DataFrame, 
    df_sources: pd.DataFrame, 
    df_evidence: pd.DataFrame,
    df_match_summary: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Se consolidan las características métricas y cualitativas obtenidas de OSINT a nivel de entidad.
    Se integra la información proveniente del resumen de coincidencias para conformar la matriz de atributos.
    """
    logger.info("Consolidación cuantitativa por entidad en progreso.")
    
    # 1. Agrupaciones sobre entity_source_results.csv
    sources_eval = df_sources.groupby('entity_id').size().rename('sources_evaluated')
    sources_hallazgo = df_sources[df_sources['evidence_count'] > 0].groupby('entity_id').size().rename('sources_with_hallazgo')
    
    # 2. Agrupaciones sobre evidence_items.csv
    max_id_score = df_evidence.groupby('entity_id')['identity_score'].max().rename('max_identity_score')
    evidence_cnt = df_evidence.groupby('entity_id').size().rename('evidence_items')
    
    df_evidence['is_review'] = df_evidence['review_required'].astype(str).str.lower().isin(['true', '1'])
    review_cnt = df_evidence[df_evidence['is_review']].groupby('entity_id').size().rename('review_items')
    
    # Se realiza la combinación con el catálogo
    df = df_entities.merge(sources_eval, on='entity_id', how='left')
    df = df.merge(sources_hallazgo, on='entity_id', how='left')
    df = df.merge(max_id_score, on='entity_id', how='left')
    df = df.merge(evidence_cnt, on='entity_id', how='left')
    df = df.merge(review_cnt, on='entity_id', how='left')
    
    # Se agregan atributos de resumen de coincidencia
    if df_match_summary is not None and not df_match_summary.empty:
        df_match_sub = df_match_summary[['entity_id', 'match_count', 'sources_hit']].copy()
        df = df.merge(df_match_sub, on='entity_id', how='left')
        df['match_count'] = df['match_count'].fillna(0).astype(int)
        df['sources_hit'] = df['sources_hit'].fillna("")
    else:
        df['match_count'] = df['evidence_items']
        df['sources_hit'] = ""
    
    # Se imputan valores nulos por defecto de agrupación
    df['sources_evaluated'] = df['sources_evaluated'].fillna(0).astype(int)
    df['sources_with_hallazgo'] = df['sources_with_hallazgo'].fillna(0).astype(int)
    df['max_identity_score'] = df['max_identity_score'].fillna(0.0)
    df['evidence_items'] = df['evidence_items'].fillna(0).astype(int)
    df['review_items'] = df['review_items'].fillna(0).astype(int)
    
    # 3. Se define la decisión general
    df['overall_decision'] = np.where(
        df['review_items'] > 0,
        'needs_review',
        np.where(df['evidence_items'] > 0, 'accepted', 'no_match')
    )

    # 4. Se verifica y carga la decisión real del analista si está disponible (MLOps Best Practice)
    if df_match_summary is not None and 'is_suspicious_analyst' in df_match_summary.columns:
        logger.info("Carga de etiquetas del analista (is_suspicious_analyst) desde la data cruda.")
        df_label = df_match_summary[['entity_id', 'is_suspicious_analyst']].copy()
        df = df.merge(df_label, on='entity_id', how='left')
        df['is_suspicious_analyst'] = df['is_suspicious_analyst'].fillna(0).astype(int)
    else:
        logger.warning("Sin etiquetas reales del analista. Flujo configurado en modo no supervisado.")
    logger.info("Consolidación finalizada.")
    if 'is_suspicious_analyst' in df.columns:
        logger.info(f"Distribución de etiqueta del analista:\n"
                    f"{df['is_suspicious_analyst'].value_counts(normalize=True).round(4)}")
        
    return df

def build_relational_graph_data(
    df_entities: pd.DataFrame, 
    df_evidence: pd.DataFrame
) -> pd.DataFrame:
    """
    Se construyen las relaciones implícitas de identidad y de índole financiera entre las entidades.
    Se establece un enlace entre entidades cuando comparten referencias URL, hashes de contenido
    o identificadores comunes dentro del conjunto de evidencias recolectadas.
    """
    logger.info("Construcción de relaciones implícitas en progreso.")
    edges = []

    # 1. Se asocian entidades que comparten referencia URL
    if 'url_or_reference' in df_evidence.columns and 'entity_id' in df_evidence.columns:
        url_groups = df_evidence.dropna(subset=['url_or_reference', 'entity_id']).groupby('url_or_reference')
        for url, group in url_groups:
            entities_list = group['entity_id'].unique().tolist()
            if len(entities_list) > 1:
                for i in range(len(entities_list)):
                    for j in range(i + 1, len(entities_list)):
                        edges.append({
                            'source': entities_list[i],
                            'target': entities_list[j],
                            'relation_type': 'shared_reference',
                            'weight': 1.0
                        })

    # 2. Se asocian entidades que comparten el mismo hash de contenido (raw_content_hash)
    if 'raw_content_hash' in df_evidence.columns and 'entity_id' in df_evidence.columns:
        hash_groups = df_evidence.dropna(subset=['raw_content_hash', 'entity_id']).groupby('raw_content_hash')
        for content_hash, group in hash_groups:
            entities_list = group['entity_id'].unique().tolist()
            if len(entities_list) > 1:
                for i in range(len(entities_list)):
                    for j in range(i + 1, len(entities_list)):
                        edges.append({
                            'source': entities_list[i],
                            'target': entities_list[j],
                            'relation_type': 'shared_content',
                            'weight': 1.5
                        })

    # Se convierte a DataFrame y se consolidan las aristas acumulando el peso
    if not edges:
        logger.warning("Sin relaciones directas detectadas.")
        return pd.DataFrame(columns=['source', 'target', 'relation_type', 'weight'])

    df_edges = pd.DataFrame(edges)
    df_edges = df_edges.groupby(['source', 'target', 'relation_type'], as_index=False)['weight'].sum()
    logger.info(f"Grafo construido. Aristas únicas: {len(df_edges)}")
    return df_edges

def prepare_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Se ejecuta el flujo completo del procesamiento de datos, abarcando las etapas de carga,
    catalogación, agregación y generación del grafo relacional.
    """
    df_sources, df_evidence, df_match_summary = load_raw_data()
    df_entities = parse_entity_catalog(df_sources, df_evidence, df_match_summary)
    
    df_consolidated = consolidate_entity_features(df_entities, df_sources, df_evidence, df_match_summary)
    df_edges = build_relational_graph_data(df_entities, df_evidence)
    
    # Se guardan los resultados en el directorio de ejecución de la corrida
    df_consolidated.to_csv(RUN_DIR / 'consolidated_entities.csv', index=False)
    df_edges.to_csv(RUN_DIR / 'entity_edges.csv', index=False)
    df_evidence.to_csv(RUN_DIR / 'processed_evidence_items.csv', index=False)
    
    logger.info(f"Archivos guardados en: {RUN_DIR}")
    return df_consolidated, df_edges, df_evidence

if __name__ == '__main__':
    prepare_pipeline()
