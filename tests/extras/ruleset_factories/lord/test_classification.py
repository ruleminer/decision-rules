import json
import os
from typing import List, Tuple
from unittest import TestCase

from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.ruleset_factories._factories.classification import LordRuleSetFactory
from tests.loaders import (
    load_dataset_to_x_y,
    load_ruleset_factories_resources_path,
)


class ClassificationLordTest(TestCase):
    """
    Test the LordRuleSetFactory class for classification rulesets.
    """

    def _load_rules(self, dataset: str) -> Tuple[List[str], dict]:
        """
        Load LORD rules (lines) and expected rules info (in JSON) for a given dataset.
        
        We assume that, for example, we have:
          - {dataset}_LORD.txt      -> the LORD output lines
          - {dataset}_factory_output.json -> JSON with expected info (e.g., number of rules)
        """
        rules_dir = load_ruleset_factories_resources_path()

        lord_txt_path = os.path.join(rules_dir, f"{dataset}_LORD.txt")
        with open(lord_txt_path, "r", encoding="utf-8") as f:
            lord_rules_lines = f.readlines()

        expected_json_path = os.path.join(rules_dir, f"{dataset}_lord_factory_output.json")
        with open(expected_json_path, "r", encoding="utf-8") as f:
            expected_info = json.load(f)

        return lord_rules_lines, expected_info

    def test_classification_iris(self):
        """
        Example test with 'iris' dataset.
        Checks if the number of rules in the resulting RuleSet
        matches the expected JSON data.
        """
        lord_rules_lines, expected_info = self._load_rules("iris")
        X, y = load_dataset_to_x_y("iris.csv")
        
        ruleset: ClassificationRuleSet = LordRuleSetFactory().make(
            lord_rules_lines,
            X,
            y,
            measure_name="C2"
        )

        self.assertEqual(len(ruleset.rules), len(expected_info["rules"]))
        for i, rule_obj in enumerate(ruleset.rules):
            hv = expected_info["rules"][i].get("voting_weight", None)
            self.assertAlmostEqual(rule_obj.voting_weight, float(hv), places=6)


    def test_classification_credit(self):
        """
        Another dataset: 'credit'
        """
        lord_rules_lines, expected_info = self._load_rules("credit")
        X, y = load_dataset_to_x_y("classification/credit.csv")

        ruleset: ClassificationRuleSet = LordRuleSetFactory().make(
            lord_rules_lines,
            X,
            y,
            measure_name="Correlation"
        )

        self.assertEqual(len(ruleset.rules), len(expected_info["rules"]))
        for i, rule_obj in enumerate(ruleset.rules):
            hv = expected_info["rules"][i].get("voting_weight", None)
            self.assertAlmostEqual(rule_obj.voting_weight, float(hv), places=6)
