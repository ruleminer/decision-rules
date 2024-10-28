from decision_rules.similarity import calculate_ruleset_similarity
from tests.base_tests.similarity.base import BaseSimilarityTest


class WholeRulesetSimilarityTest(BaseSimilarityTest):
    def test_ruleset_similarity(self):
        similarity = calculate_ruleset_similarity(
            self.ruleset1, self.ruleset2, self.dataset)
        self.assertEqual(similarity, 1.0)
