import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import DATA_PROCESSED_DIR
from src.data_processing import parse_entity_catalog, consolidate_entity_features
from src.embeddings_clustering import calculate_cohesion

class TestAMLPipeline(unittest.TestCase):
    """
    Se definen las pruebas unitarias para validar la integridad de la lógica de negocio
    y el procesamiento de datos del sistema de cumplimiento PLD/AML.
    """

    def setUp(self):
        """
        Se realiza la preparación de los datos mínimos de prueba para la ejecución de las aserciones.
        """
        self.df_sources_mock = pd.DataFrame([
            {
                'entity_id': 'OFAC-01',
                'query_used': "[{'entity_id': 'OFAC-01', 'query_value': 'JUAN PEREZ', 'metadata': {'country_code': 'MEX', 'entity_type': 'PF'}}]",
                'evidence_count': 1
            },
            {
                'entity_id': 'OFAC-02',
                'query_used': "[{'entity_id': 'OFAC-02', 'query_value': 'EMPRESA SA', 'metadata': {'country_code': 'PAN', 'entity_type': 'PM'}}]",
                'evidence_count': 0
            }
        ])

        self.df_evidence_mock = pd.DataFrame([
            {
                'entity_id': 'OFAC-01',
                'identity_score': 0.9,
                'review_required': 'True'
            },
            {
                'entity_id': 'OFAC-01',
                'identity_score': 0.8,
                'review_required': 'False'
            }
        ])

        self.df_match_mock = pd.DataFrame([
            {
                'entity_id': 'OFAC-01',
                'entity_name': 'JUAN PEREZ',
                'input_country': 'MEX',
                'input_entity_type': 'PF',
                'sources_hit': 'OFAC_SDN',
                'match_count': 2,
                'matches_json': '[]',
                'is_suspicious_analyst': 1
            }
        ])

    def test_entity_catalog_parsing(self):
        """
        Se verifica que el catálogo de entidades extraiga y normalice los registros correctamente
        desde la representación serializada de query_used.
        """
        df_entities = parse_entity_catalog(self.df_sources_mock, self.df_evidence_mock, self.df_match_mock)
        self.assertEqual(len(df_entities), 2)
        self.assertIn('entity_id', df_entities.columns)
        self.assertIn('entity_name', df_entities.columns)
        self.assertEqual(df_entities.loc[df_entities['entity_id'] == 'OFAC-01', 'entity_type'].values[0], 'FISICA')
        self.assertEqual(df_entities.loc[df_entities['entity_id'] == 'OFAC-02', 'entity_type'].values[0], 'MORAL')

    def test_entity_feature_consolidation(self):
        """
        Se verifica que la agregación cuantitativa a nivel de entidad se consolide adecuadamente,
        así como el cálculo de la decisión general del pipeline.
        """
        df_entities = parse_entity_catalog(self.df_sources_mock, self.df_evidence_mock, self.df_match_mock)
        df_consolidated = consolidate_entity_features(df_entities, self.df_sources_mock, self.df_evidence_mock, self.df_match_mock)
        
        # Validar registros agregados
        self.assertEqual(df_consolidated.loc[df_consolidated['entity_id'] == 'OFAC-01', 'evidence_items'].values[0], 2)
        self.assertEqual(df_consolidated.loc[df_consolidated['entity_id'] == 'OFAC-01', 'max_identity_score'].values[0], 0.9)
        self.assertEqual(df_consolidated.loc[df_consolidated['entity_id'] == 'OFAC-01', 'overall_decision'].values[0], 'needs_review')
        self.assertEqual(df_consolidated.loc[df_consolidated['entity_id'] == 'OFAC-02', 'overall_decision'].values[0], 'no_match')
        self.assertEqual(df_consolidated.loc[df_consolidated['entity_id'] == 'OFAC-01', 'match_count'].values[0], 2)
        self.assertEqual(df_consolidated.loc[df_consolidated['entity_id'] == 'OFAC-01', 'sources_hit'].values[0], 'OFAC_SDN')
        self.assertEqual(df_consolidated.loc[df_consolidated['entity_id'] == 'OFAC-01', 'is_suspicious_analyst'].values[0], 1)
        self.assertEqual(df_consolidated.loc[df_consolidated['entity_id'] == 'OFAC-02', 'is_suspicious_analyst'].values[0], 0)

    def test_calculate_cohesion_metric(self):
        """
        Se verifica que la cohesión de clústeres retorne valores esperados para distancias simuladas.
        """
        X = np.array([[1.0, 1.0], [1.1, 0.9], [5.0, 5.0], [5.2, 4.8]])
        labels = np.array([0, 0, 1, 1])
        cohesion = calculate_cohesion(X, labels)
        
        self.assertTrue(cohesion > 0.0)
        self.assertTrue(cohesion < 1.0)

    def test_pipeline_modes_and_registry(self):
        """
        Se valida que los directorios del registro compartido y las corridas por modo
        se definan correctamente de acuerdo a las variables del sistema.
        """
        from src.config import SHARED_DIR, TRAIN_RUNS_DIR, USE_RUNS_DIR
        
        self.assertTrue(SHARED_DIR.exists())
        self.assertTrue((SHARED_DIR / "models").exists())
        self.assertTrue(TRAIN_RUNS_DIR.exists())
        self.assertTrue(USE_RUNS_DIR.exists())

if __name__ == '__main__':
    unittest.main()
