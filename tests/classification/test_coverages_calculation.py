# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import json
import os
import unittest

import numpy as np
import pandas as pd
from decision_rules.classification.rule import ClassificationRule
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.serialization import JSONSerializer


class TestClassificationCoveragesCalculation(unittest.TestCase):

    def _prepare_test_rule(self) -> ClassificationRule:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ruleset_file_path: str = os.path.join(
            dir_path, '..', 'resources', 'iris_ruleset.json')

        with open(ruleset_file_path, 'r', encoding='utf-8') as file:
            return JSONSerializer.deserialize(
                json.load(file),
                ClassificationRuleSet
            )

    def _prepare_test_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dataset_path: str = os.path.join(
            dir_path, '..', 'resources', 'iris.csv')

        df = pd.read_csv(dataset_path)
        X = df.drop('class', axis=1)
        y = df['class']
        return X, y

    def test_calculate_metrics_without_calculated_coverage(self):
        ruleset: ClassificationRuleSet = self._prepare_test_rule()
        X, y = self._prepare_test_dataset()

        ruleset.calculate_rules_coverages(X, y)
        self.assertIsNotNone(ruleset)

    # Test for ROLAP-635 issue
    def test_calculate_metrics_with_calculated_coverage(self):
        """Test coverage calculation for dataset where all rows has the same class
        """
        ruleset: ClassificationRuleSet = self._prepare_test_rule()
        X, y = self._prepare_test_dataset()
        y_replaced = pd.Series(np.repeat(y.unique()[0], y.shape[0]))
        ruleset.calculate_rules_coverages(X, y_replaced)


if __name__ == '__main__':
    unittest.main()
