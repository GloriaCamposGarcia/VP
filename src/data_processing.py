import ast
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from src.config import logger, DATA_RAW_DIR, DATA_PROCESSED_DIR, RUN_DIR, PIPELINE_MODE

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los datasets de resultados de fuentes, evidencias y resumen de coincidencia
    desde la carpeta de datos crudos, con soporte para subcarpetas de entrenamiento/uso.
    """
    raw_dir = DATA_RAW_DIR / PIPELINE_MODE if (DATA_RAW_DIR / PIPELINE_MODE).exists() else DATA_RAW_DIR
    logger.info(f"Cargando archivos crudos desde el directorio: {raw_dir}")

    source_results_path = raw_dir / 'entity_source_results.csv'
    evidence_items_path = raw_dir / 'evidence_items.csv'
    match_summary_path = raw_dir / 'entity_match_summary.csv'

    if not source_results_path.exists() or not evidence_items_path.exists() or not match_summary_path.exists():
        raise FileNotFoundError(
            f"Falta alguno de los datasets necesarios en {raw_dir}."
        )

    logger.info("Cargando datasets crudos de AML/OSINT...")
    df_sources = pd.read_csv(source_results_path)
    df_evidence = pd.read_csv(evidence_items_path)
    df_match_summary = pd.read_csv(match_summary_path)
    logger.info(
        f"Resultados de fuentes: {df_sources.shape}, "
        f"Ítems de evidencia: {df_evidence.shape}, "
        f"Resumen de coincidencias: {df_match_summary.shape}"
    )
    return df_sources, df_evidence, df_match_summary

def parse_entity_catalog(
    df_sources: pd.DataFrame, 
    df_evidence: pd.DataFrame = None, 
    df_match_summary: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Extrae un catálogo unificado de entidades únicas. Prioriza el uso de 
    df_match_summary. Si faltan entidades en él, se recurre a la columna 
    'query_used' de df_sources y finalmente a df_evidence.
    """
    logger.info("Construyendo el catálogo de entidades únicas...")
    entities: Dict[str, dict] = {}
    
    # Diccionario para mapear nombres de países a códigos de 3 letras
    country_mapping = {
        'mexico': 'MEX', 'cuba': 'CUB', 'panama': 'PAN', 
        'spain': 'ESP', 'united kingdom': 'GBR', 'switzerland': 'CHE',
        'france': 'FRA', 'italy': 'ITA', 'argentina': 'ARG',
        'china': 'CHN', 'canada': 'CAN', 'united states': 'USA',
        'hong kong': 'HKG'
    }

    # 1. Priorizar información de df_match_summary
    if df_match_summary is not None and not df_match_summary.empty:
        logger.info("Cargando metadatos base desde df_match_summary...")
        for _, row in df_match_summary.iterrows():
            ent_id = row['entity_id']
            name = row['entity_name']
            raw_type = row['input_entity_type']
            country = row['input_country']
            
            # Normalizar tipo de entidad según la práctica CNBV (Persona Física / Moral)
            ent_type = 'MORAL' if raw_type == 'PM' else ('FISICA' if raw_type == 'PF' else 'OTRO')
            
            # Normalizar país
            country_str = str(country).strip() if pd.notna(country) else ''
            if country_str.lower() in ['unknown', '']:
                country_code = 'DESCONOCIDO'
            else:
                country_code = country_mapping.get(country_str.lower(), country_str.upper())
                
            if ent_id and name and ent_id not in entities:
                entities[ent_id] = {
                    'entity_id': ent_id,
                    'entity_name': name,
                    'entity_type': ent_type,
                    'country_code': country_code
                }

    # 2. Completar con df_sources ('query_used') para entidades que no tengan matches
    logger.info("Buscando entidades adicionales en query_used de df_sources...")
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
                        'country_code': country if country else 'DESCONOCIDO'
                    }
        except Exception as e:
            continue
            
    df_entities = pd.DataFrame(list(entities.values()))
    
    # 3. Si alguna entidad queda fuera de las anteriores (por seguridad), resolver con df_evidence
    unique_ids_sources = df_sources['entity_id'].dropna().unique()
    missing_ids = set(unique_ids_sources) - set(df_entities['entity_id']) if not df_entities.empty else set(unique_ids_sources)
    if missing_ids:
        logger.info(f"Resolviendo país real para {len(missing_ids)} entidades sin metadatos en resumen ni query_used...")
        missing_rows = []
        
        for mid in missing_ids:
            detected_country = 'DESCONOCIDO'
            
            if df_evidence is not None and not df_evidence.empty and 'country' in df_evidence.columns:
                sub_ev = df_evidence[df_evidence['entity_id'] == mid]
                if not sub_ev.empty:
                    raw_val = str(sub_ev['country'].iloc[0]).strip()
                    if raw_val:
                        normalized_val = raw_val.lower()
                        detected_country = country_mapping.get(normalized_val, raw_val[:3].upper())
            
            missing_rows.append({
                'entity_id': mid,
                'entity_name': f"Entidad Desconocida {mid}",
                'entity_type': 'OTRO',
                'country_code': detected_country
            })
        df_entities = pd.concat([df_entities, pd.DataFrame(missing_rows)], ignore_index=True)

    logger.info(f"Catálogo de entidades construido con {len(df_entities)} entidades únicas.")
    return df_entities

def consolidate_entity_features(
    df_entities: pd.DataFrame, 
    df_sources: pd.DataFrame, 
    df_evidence: pd.DataFrame,
    df_match_summary: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Consolida las métricas cuantitativas y cualitativas de OSINT a nivel de entidad,
    incorporando información consolidada de resumen de coincidencias.
    """
    logger.info("Consolidando características cuantitativas por entidad...")
    
    # 1. Agrupaciones sobre entity_source_results.csv
    sources_eval = df_sources.groupby('entity_id').size().rename('sources_evaluated')
    sources_hallazgo = df_sources[df_sources['evidence_count'] > 0].groupby('entity_id').size().rename('sources_with_hallazgo')
    
    # 2. Agrupaciones sobre evidence_items.csv
    max_id_score = df_evidence.groupby('entity_id')['identity_score'].max().rename('max_identity_score')
    evidence_cnt = df_evidence.groupby('entity_id').size().rename('evidence_items')
    
    df_evidence['is_review'] = df_evidence['review_required'].astype(str).str.lower().isin(['true', '1'])
    review_cnt = df_evidence[df_evidence['is_review']].groupby('entity_id').size().rename('review_items')
    
    # Combinar con el catálogo
    df = df_entities.merge(sources_eval, on='entity_id', how='left')
    df = df.merge(sources_hallazgo, on='entity_id', how='left')
    df = df.merge(max_id_score, on='entity_id', how='left')
    df = df.merge(evidence_cnt, on='entity_id', how='left')
    df = df.merge(review_cnt, on='entity_id', how='left')
    
    # Agregar columnas de df_match_summary
    if df_match_summary is not None and not df_match_summary.empty:
        df_match_sub = df_match_summary[['entity_id', 'match_count', 'sources_hit']].copy()
        df = df.merge(df_match_sub, on='entity_id', how='left')
        df['match_count'] = df['match_count'].fillna(0).astype(int)
        df['sources_hit'] = df['sources_hit'].fillna("")
    else:
        df['match_count'] = df['evidence_items']
        df['sources_hit'] = ""
    
    # Llenar nulos por defecto de agrupación
    df['sources_evaluated'] = df['sources_evaluated'].fillna(0).astype(int)
    df['sources_with_hallazgo'] = df['sources_with_hallazgo'].fillna(0).astype(int)
    df['max_identity_score'] = df['max_identity_score'].fillna(0.0)
    df['evidence_items'] = df['evidence_items'].fillna(0).astype(int)
    df['review_items'] = df['review_items'].fillna(0).astype(int)
    
    # 3. overall_decision del pipeline
    df['overall_decision'] = np.where(
        df['review_items'] > 0,
        'needs_review',
        np.where(df['evidence_items'] > 0, 'accepted', 'no_match')
    )

    # 4. Simular etiqueta del analista Human-in-the-Loop (is_suspicious_analyst)
    # El analista marcará como sospechoso real (is_suspicious_analyst = 1)
    # a aquellos casos que requieren revisión (needs_review) y que tienen evidencia sustancial
    # (más de 2 fuentes con hallazgo o un score de coincidencia de identidad alto o múltiples evidencias).
    # De lo contrario (ej. homónimo con bajo score de identidad o sin evidencias reales), se marca como 0.
    q75_identity = df['max_identity_score'].quantile(0.75) if len(df) > 0 else 0.8
    
    df['is_suspicious_analyst'] = (
        (df['overall_decision'] == 'needs_review') & (
            (df['sources_with_hallazgo'] >= 2)
            | (df['max_identity_score'] >= q75_identity)
            | (df['evidence_items'] >= 3)
        )
    ).astype(int)
    
    logger.info("Consolidación finalizada.")
    logger.info(f"Distribución de la etiqueta simulada del analista (is_suspicious_analyst):\n"
                f"{df['is_suspicious_analyst'].value_counts(normalize=True).round(4)}")
    return df

def build_relational_graph_data(
    df_entities: pd.DataFrame, 
    df_evidence: pd.DataFrame
) -> pd.DataFrame:
    """
    Construye relaciones financieras/identidad implícitas entre entidades de manera no sintética.
    Conecta entidades si:
    - Comparten el mismo 'url_or_reference' (investigadas en la misma lista/noticia).
    - Comparten el mismo 'snippet_hash' o 'raw_content_hash'.
    - Comparten identificadores dentro de la columna 'identifiers'.
    """
    logger.info("Construyendo relaciones implícitas entre entidades para análisis de grafos...")
    edges = []

    # 1. Comparten el mismo url de referencia
    if 'url_or_reference' in df_evidence.columns and 'entity_id' in df_evidence.columns:
        url_groups = df_evidence.dropna(subset=['url_or_reference', 'entity_id']).groupby('url_or_reference')
        for url, group in url_groups:
            entities_list = group['entity_id'].unique().tolist()
            if len(entities_list) > 1:
                # Crear aristas entre todas las entidades que comparten esta referencia
                for i in range(len(entities_list)):
                    for j in range(i + 1, len(entities_list)):
                        edges.append({
                            'source': entities_list[i],
                            'target': entities_list[j],
                            'relation_type': 'shared_reference',
                            'weight': 1.0
                        })

    # 2. Comparten el mismo raw_content_hash
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

    # Convertir a dataframe y consolidar aristas repetidas sumando pesos
    if not edges:
        logger.warning("No se detectaron relaciones directas en el dataset de evidencias.")
        return pd.DataFrame(columns=['source', 'target', 'relation_type', 'weight'])

    df_edges = pd.DataFrame(edges)
    df_edges = df_edges.groupby(['source', 'target', 'relation_type'], as_index=False)['weight'].sum()
    logger.info(f"Grafo relacional construido con {len(df_edges)} aristas únicas reales.")
    return df_edges

def prepare_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el pipeline completo de procesamiento de datos:
    carga, cataloga, agrupa y genera el grafo.
    """
    df_sources, df_evidence, df_match_summary = load_raw_data()
    df_entities = parse_entity_catalog(df_sources, df_evidence, df_match_summary)
    
    # Consolidar entidad
    df_consolidated = consolidate_entity_features(df_entities, df_sources, df_evidence, df_match_summary)
    
    # Crear grafo
    df_edges = build_relational_graph_data(df_entities, df_evidence)
    
    # Guardar en processed
    df_consolidated.to_csv(RUN_DIR / 'consolidated_entities.csv', index=False)
    df_edges.to_csv(RUN_DIR / 'entity_edges.csv', index=False)
    df_evidence.to_csv(RUN_DIR / 'processed_evidence_items.csv', index=False)
    
    logger.info(f"Archivos guardados exitosamente en {RUN_DIR}")
    return df_consolidated, df_edges, df_evidence

if __name__ == '__main__':
    prepare_pipeline()
