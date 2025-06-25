# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import json
import os
import unittest

import numpy as np
import pandas as pd

from decision_rules import measures
from decision_rules.problem import ProblemTypes
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.serialization.utils import JSONSerializer
from tests.loaders import load_dataset, load_resources_path, load_ruleset


class TestRegressionRuleSetImportanceCalculation(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        df = pd.read_csv(os.path.join(load_resources_path(), "regression", "bolts.csv"))
        self.X = df.drop("class", axis=1)
        self.y = df["class"].replace("?", np.nan).astype(float)

        ruleset_file_path: str = os.path.join(
            load_resources_path(), "regression", "bolts_ruleset.json"
        )
        with open(ruleset_file_path, "r", encoding="utf-8") as file:
            self.ruleset: RegressionRuleSet = JSONSerializer.deserialize(
                json.load(file), RegressionRuleSet
            )

    def test_condition_importances(self):
        """Test ROLAP-1214 issue, where calculation of the condition importances modifies
        rules conclusion values
        """
        original_conclusions_values: list[float] = [
            JSONSerializer.serialize(rule.conclusion) for rule in self.ruleset.rules
        ]
        self.ruleset.calculate_condition_importances(
            self.X, self.y, measures.correlation
        )

        current_conclusions_values: list[float] = [
            JSONSerializer.serialize(rule.conclusion) for rule in self.ruleset.rules
        ]
        self.assertEqual(
            original_conclusions_values,
            current_conclusions_values,
            "All rules conclusions should remain unchanged",
        )

    def test_on_deeprules(self):
        """
        Test for Issue #41 (https://github.com/ruleminer/decision-rules/issues/41)
        """
        self.ruleset: RegressionRuleSet = load_ruleset(
            "regression/boston_deeprules.json", ProblemTypes.REGRESSION
        )
        df = load_dataset("regression/boston.csv")
        self.X, self.y = self.ruleset.split_dataset(df)
        condition_importances = self.ruleset.calculate_condition_importances(
            self.X, self.y, measure=measures.c2
        )
        _ = self.ruleset.calculate_attribute_importances(
            condition_importances
        )


if __name__ == "__main__":
    unittest.main()
