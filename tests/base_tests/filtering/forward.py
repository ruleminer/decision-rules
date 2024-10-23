from decision_rules.filtering import FilterAlgorithm
from decision_rules.measures import c2
from tests.filtering.base import TestRulesetFiltering


class TestRulesetFilteringForward(TestRulesetFiltering):
    def test_filter_ruleset_classification_forward(self):
        self.base_test_filter_ruleset(
            self.classification_ruleset, self.classification_dataset,
            FilterAlgorithm.Forward, 0.1, c2
        )

    def test_filter_ruleset_regression_forward(self):
        self.base_test_filter_ruleset(
            self.regression_ruleset, self.regression_dataset,
            FilterAlgorithm.Forward, 0.2, "c2"
        )

    def test_filter_ruleset_survival_forward(self):
        self.base_test_filter_ruleset(
            self.survival_ruleset, self.survival_dataset,
            FilterAlgorithm.Forward, 0.0, None
        )

    def test_filter_ruleset_iris_forward(self):
        self.base_test_filter_ruleset(
            self.iris_ruleset, self.iris_dataset,
            FilterAlgorithm.Forward, 0.1, "c2"
        )
