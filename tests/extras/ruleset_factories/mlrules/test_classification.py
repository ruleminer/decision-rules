import json
import os
from typing import Tuple, List
from unittest import TestCase

from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.ruleset_factories._factories.classification import (
    MLRulesRuleSetFactory,
)

from tests.loaders import (
    load_dataset_to_x_y,
    load_ruleset_factories_resources_path,
)


class ClassificationMLRulesTest(TestCase):
    """
    Test the MLRulesRuleSetFactory class for classification rulesets.
    The tests are performed on three datasets:
        - "credit",
        - "iris",
        - "titanic".
    """

    def _load_rules(self, dataset: str) -> Tuple[List[str], dict]:
        """Load MLRules and expected rules from files for given dataset."""
        rules_dir = load_ruleset_factories_resources_path()
        with open(os.path.join(rules_dir, f"{dataset}_MLRules.txt")) as file:
            ml_rules_lines = file.readlines()
        with open(os.path.join(rules_dir, f"{dataset}_factory_output.json")) as file:
            expected_rules = json.load(file)
        return ml_rules_lines, expected_rules

    def test_classification_credit(self):
        ml_rules_lines, expected_rules = self._load_rules("credit")
        X, y = load_dataset_to_x_y("classification/credit.csv")
        ruleset: ClassificationRuleSet = MLRulesRuleSetFactory().make(
            ml_rules_lines, X, y, "Precision"
        )
        self.assertEqual(len(ruleset.rules), len(expected_rules["rules"]))

    def test_classification_iris(self):
        ml_rules_lines, expected_rules = self._load_rules("iris")
        X, y = load_dataset_to_x_y("iris.csv")
        ruleset: ClassificationRuleSet = MLRulesRuleSetFactory().make(
            ml_rules_lines, X, y, "C2"
        )
        self.assertEqual(len(ruleset.rules), len(expected_rules["rules"]))

    def test_classification_titanic(self):
        ml_rules_lines, expected_rules = self._load_rules("titanic")
        X, y = load_dataset_to_x_y("classification/titanic.csv")
        ruleset: ClassificationRuleSet = MLRulesRuleSetFactory().make(
            ml_rules_lines, X, y, "Correlation"
        )
        self.assertEqual(len(ruleset.rules), len(expected_rules["rules"]))
