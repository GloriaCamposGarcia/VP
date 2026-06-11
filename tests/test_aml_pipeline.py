import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import DATA_PROCESSED_DIR
from src.data_processing import parse_entity_catalog, consolidate_entity_features
from src.embeddings_clustering import calculate_cohesion

class TestAMLPipeline(unittest.TestCase):
    """
    Define las pruebas unitarias para validar la integridad de la lógica de negocio
    y el procesamiento de datos del sistema de cumplimiento PLD/AML.
    """

    def setUp(self):
        """
        Prepara los datos mínimos de prueba para la ejecución de las aserciones.
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

    def test_entity_catalog_parsing(self):
        """
        Verifica que el catálogo de entidades extraiga y normalice los registros correctamente
        desde la representación serializada de query_used.
        """
        df_entities = parse_entity_catalog(self.df_sources_mock)
        self.assertEqual(len(df_entities), 2)
        self.assertIn('entity_id', df_entities.columns)
        self.assertIn('entity_name', df_entities.columns)
        self.assertEqual(df_entities.loc[df_entities['entity_id'] == 'OFAC-01', 'entity_type'].values[0], 'FISICA')
        self.assertEqual(df_entities.loc[df_entities['entity_id'] == 'OFAC-02', 'entity_type'].values[0], 'MORAL')

    def test_entity_feature_consolidation(self):
        """
        Verifica que la agregación cuantitativa a nivel entidad se consolide adecuadamente,
        así como el cálculo de la decisión general.
        """
        df_entities = parse_entity_catalog(self.df_sources_mock)
        df_consolidated = consolidate_entity_features(df_entities, self.df_sources_mock, self.df_evidence_mock)
        
        # Validar registros agregados
        self.assertEqual(df_consolidated.loc[df_consolidated['entity_id'] == 'OFAC-01', 'evidence_items'].values[0], 2)
        self.assertEqual(df_consolidated.loc[df_consolidated['entity_id'] == 'OFAC-01', 'max_identity_score'].values[0], 0.9)
        self.assertEqual(df_consolidated.loc[df_consolidated['entity_id'] == 'OFAC-01', 'overall_decision'].values[0], 'needs_review')
        self.assertEqual(df_consolidated.loc[df_consolidated['entity_id'] == 'OFAC-02', 'overall_decision'].values[0], 'no_match')

    def test_calculate_cohesion_metric(self):
        """
        Verifica que la cohesión de clústeres retorne valores esperados para distancias simuladas.
        """
        # Crear matriz de puntos donde la cohesión es calculable
        X = np.array([[1.0, 1.0], [1.1, 0.9], [5.0, 5.0], [5.2, 4.8]])
        labels = np.array([0, 0, 1, 1])
        cohesion = calculate_cohesion(X, labels)
        
        # Debe retornar un número flotante válido y menor que la distancia inter-clúster
        self.assertTrue(cohesion > 0.0)
        self.assertTrue(cohesion < 1.0)

if __name__ == '__main__':
    unittest.main()
