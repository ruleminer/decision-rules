from decision_rules.filtering import FilterAlgorithm
from decision_rules.measures import accuracy
from tests.filtering.base import TestRulesetFiltering


class TestRulesetFilteringBackward(TestRulesetFiltering):
    def test_filter_ruleset_classification_backward(self):
        self.base_test_filter_ruleset(
            self.classification_ruleset, self.classification_dataset,
            FilterAlgorithm.Backward, 0.1, accuracy)

    def test_filter_ruleset_regression_backward(self):
        self.base_test_filter_ruleset(
            self.regression_ruleset, self.regression_dataset,
            FilterAlgorithm.Backward, 0.2, "accuracy")

    def test_filter_ruleset_survival_backward(self):
        self.base_test_filter_ruleset(
            self.survival_ruleset, self.survival_dataset,
            FilterAlgorithm.Backward, 0.0, None)

    def test_filter_ruleset_iris_backward(self):
        self.base_test_filter_ruleset(
            self.iris_ruleset, self.iris_dataset,
            FilterAlgorithm.Backward, 0.1, "accuracy")
