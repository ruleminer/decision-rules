# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import copy
import json
import os
import unittest

import numpy as np
import pandas as pd
from decision_rules import measures
from decision_rules.regression.rule import RegressionConclusion
from decision_rules.regression.rule import RegressionRule
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.serialization.utils import JSONSerializer

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestRegressionRuleSet(unittest.TestCase):

    """
    Test for ROLAP-441 bug during prediction for regression problem
    """

    def setUp(self) -> None:
        super().setUp()
        df = pd.read_csv(os.path.join(
            dir_path, '..', 'resources', 'regression', 'diabetes.csv'
        ))
        self.X = df.drop('label', axis=1)
        self.y = df['label']

        ruleset_file_path: str = os.path.join(
            dir_path, '..', 'resources', 'regression', 'diabetes_ruleset.json')
        with open(ruleset_file_path, 'r', encoding='utf-8') as file:
            self.ruleset: RegressionRuleSet = JSONSerializer.deserialize(
                json.load(file),
                RegressionRuleSet
            )

    def test_prediction(self):
        self.ruleset.update(self.X, self.y, measure=measures.c2)
        for strategy in self.ruleset.prediction_strategies_choice.keys():
            self.ruleset.set_prediction_strategy(strategy)
            self.ruleset.predict(self.X)

    def test_prediction_using_coverage_matrix(self):
        coverage_matrix: np.ndarray = self.ruleset.update(
            self.X, self.y, measure=measures.c2
        )
        for strategy in self.ruleset.prediction_strategies_choice.keys():
            self.ruleset.set_prediction_strategy(strategy)
            pred1 = self.ruleset.predict(self.X)
            pred2 = self.ruleset.predict_using_coverage_matrix(coverage_matrix)
            self.assertTrue(np.allclose(pred1, pred2, rtol=0.0001),
                            "Predictions should be equal for both methods")

    def test_fixed_conclusion(self):
        rule: RegressionRule = self.ruleset.rules[0]
        rule.conclusion.fixed = True
        rule.conclusion.value = rule.conclusion.value - 1
        rule.conclusion.low = rule.conclusion.low - 1
        rule.conclusion.high = rule.conclusion.high - 1
        rule.premise.subconditions[0].left = 1.5
        rule.premise.subconditions[0].right = 1.9
        old_conclusion = copy.deepcopy(rule.conclusion)

        self.ruleset.update(self.X, self.y, measure=measures.c2)

        self.assertEqual(
            old_conclusion.low, rule.conclusion.low,
            'Low value should not be changed for fixed conclusion'
        )
        self.assertEqual(
            old_conclusion.high, rule.conclusion.high,
            'High value should not be changed for fixed conclusion'
        )
        self.assertEqual(
            old_conclusion.value, rule.conclusion.value,
            'Value should not be changed for fixed conclusion'
        )
        self.assertNotEqual(
            old_conclusion.train_covered_y_mean, rule.conclusion.train_covered_y_mean,
            'train_covered_y_mean value should be recalculated'
        )
        self.assertNotEqual(
            old_conclusion.train_covered_y_max, rule.conclusion.train_covered_y_max,
            'train_covered_y_max value should be recalculated'
        )
        self.assertNotEqual(
            old_conclusion.train_covered_y_min, rule.conclusion.train_covered_y_min,
            'train_covered_y_min value should be recalculated'
        )
        self.assertNotEqual(
            old_conclusion.train_covered_y_std, rule.conclusion.train_covered_y_std,
            'train_covered_y_std value should be recalculated'
        )

        rule.conclusion.fixed = False
        self.ruleset.update(self.X, self.y, measure=measures.c2)

        self.assertNotEqual(
            old_conclusion.low, rule.conclusion.low,
            'Low value should be recalculated for fixed conclusion'
        )
        self.assertNotEqual(
            old_conclusion.high, rule.conclusion.high,
            'High value should be recalculated for fixed conclusion'
        )
        self.assertNotEqual(
            old_conclusion.value, rule.conclusion.value,
            'Value should be recalculated for fixed conclusion'
        )
        self.assertNotEqual(
            old_conclusion.train_covered_y_mean, rule.conclusion.train_covered_y_mean,
            'train_covered_y_mean value should be recalculated'
        )
        self.assertNotEqual(
            old_conclusion.train_covered_y_max, rule.conclusion.train_covered_y_max,
            'train_covered_y_max value should be recalculated'
        )
        self.assertNotEqual(
            old_conclusion.train_covered_y_min, rule.conclusion.train_covered_y_min,
            'train_covered_y_min value should be recalculated'
        )
        self.assertNotEqual(
            old_conclusion.train_covered_y_std, rule.conclusion.train_covered_y_std,
            'train_covered_y_std value should be recalculated'
        )

    def test_prediction_with_empty_default_conclusion(self):
        # remove one rule to leave some example uncovered
        self.ruleset.rules = self.ruleset.rules[10:]
        self.ruleset.update(self.X, self.y, measure=measures.c2)
        self.ruleset.set_default_conclusion_enabled(False)

        prediction: np.ndarray = self.ruleset.predict(self.X)
        self.assertTrue(
            np.isnan(prediction).any(),
            'Prediction for some examples should be empty'
        )


if __name__ == '__main__':
    unittest.main()
