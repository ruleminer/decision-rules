# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import json
import os
import unittest

import numpy as np
import pandas as pd
from decision_rules import measures
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.serialization.utils import JSONSerializer
from tests.loaders import load_resources_path

class TestRegressionRuleSetImportanceCalculation(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        df = pd.read_csv(os.path.join(
            load_resources_path(), 'regression', 'bolts.csv'
        ))
        self.X = df.drop('class', axis=1)
        self.y = df['class'].replace('?', np.nan).astype(float)

        ruleset_file_path: str = os.path.join(
            load_resources_path(), 'regression', 'bolts_ruleset.json'
        )
        with open(ruleset_file_path, 'r', encoding='utf-8') as file:
            self.ruleset: RegressionRuleSet = JSONSerializer.deserialize(
                json.load(file),
                RegressionRuleSet)

    def test_condition_importances(self):
        """Test ROLAP-1214 issue, where calculation of the condition importances modifies
        rules conclusion values
        """
        original_conclusions_values: list[float] = [
            JSONSerializer.serialize(rule.conclusion)
            for rule in self.ruleset.rules
        ]
        self.ruleset.calculate_condition_importances(
            self.X, self.y, measures.correlation)

        current_conclusions_values: list[float] = [
            JSONSerializer.serialize(rule.conclusion)
            for rule in self.ruleset.rules
        ]
        self.assertEqual(
            original_conclusions_values,
            current_conclusions_values,
            'All rules conclusions should remain unchanged'
        )


if __name__ == '__main__':
    unittest.main()
