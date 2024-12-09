# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

import numpy as np

from decision_rules.classification.metrics import ClassificationRulesMetrics
from decision_rules.classification.rule import ClassificationConclusion
from decision_rules.classification.rule import ClassificationRule
from decision_rules.conditions import CompoundCondition
from decision_rules.core.coverage import Coverage
from tests.base_tests.core.test_abstract_condition import TestingCondition


class TestClassificationRule(unittest.TestCase):

    def test_calculate_coverage(self):
        y: np.ndarray = np.array([0, 1, 1, 0, 1])
        X: np.ndarray = np.array([
            [0,], [1,], [1,], [1,], [1,]
        ])
        decision = 1
        condition = TestingCondition()
        rule = ClassificationRule(
            condition,
            conclusion=ClassificationConclusion(
                value=decision,
                column_name='label'
            ),
            column_names=['a', 'b']
        )
        coverage: Coverage = rule.calculate_coverage(X, y)

        expected_p = np.count_nonzero(
            ((X[:, condition.colum_index] == 1) & (y == decision)).astype(int)
        )
        self.assertEqual(
            coverage.p, expected_p,
            'Should calculate p correctly'
        )

        expected_n = np.count_nonzero(
            ((X[:, condition.colum_index] == 1) & (y != decision)).astype(int)
        )
        self.assertEqual(
            coverage.n, expected_n,
            'Should calculate n correctly'
        )


class TestEmptyClassificationRuleMetrics(unittest.TestCase):
    def test_empty_rule_metrics(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        rule = ClassificationRule(
            CompoundCondition([]),
            ClassificationConclusion("", 'label'),
            []
        )
        rule.coverage = Coverage(3, 3, 3, 3)
        metrics_calculator = ClassificationRulesMetrics([rule])
        metrics = metrics_calculator.calculate(X, y)
        self.assertTrue(
            np.isnan(metrics[rule.uuid]["negative_predictive_value"]))


if __name__ == '__main__':
    unittest.main()
