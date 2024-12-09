from decision_rules.filtering import FilterAlgorithm
from decision_rules.measures import precision
from tests.filtering.base import TestRulesetFiltering


class TestRulesetFilteringCoverage(TestRulesetFiltering):
    def test_filter_ruleset_classification_coverage(self):
        self.base_test_filter_ruleset(
            self.classification_ruleset, self.classification_dataset,
            FilterAlgorithm.Coverage, None, precision)

    def test_filter_ruleset_regression_coverage(self):
        self.base_test_filter_ruleset(
            self.regression_ruleset, self.regression_dataset,
            FilterAlgorithm.Coverage, None, "precision")

    def test_filter_ruleset_survival_coverage(self):
        self.base_test_filter_ruleset(
            self.survival_ruleset, self.survival_dataset,
            FilterAlgorithm.Coverage, None, None)

    def test_filter_ruleset_iris_coverage(self):
        self.base_test_filter_ruleset(
            self.iris_ruleset, self.iris_dataset,
            FilterAlgorithm.Coverage, None, "precision")
