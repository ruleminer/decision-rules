import numpy as np

from decision_rules.similarity import calculate_rule_similarity
from decision_rules.similarity import SimilarityMeasure
from decision_rules.similarity import SimilarityType
from tests.base_tests.similarity.base import BaseSimilarityTest


class SemanticRulesetSimilarityTest(BaseSimilarityTest):
    def test_semantic_similarity(self):
        similarity_matrix = calculate_rule_similarity(
            self.ruleset1, self.ruleset2, self.dataset, SimilarityType.SEMANTIC, SimilarityMeasure.JACCARD
        )
        number_of_rules = len(self.ruleset1.rules)
        expected_diagonal = np.ones(number_of_rules)
        self.assertTrue(np.allclose(
            similarity_matrix.diagonal(), expected_diagonal, atol=1e-8))
