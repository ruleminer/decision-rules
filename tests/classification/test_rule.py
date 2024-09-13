# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

import numpy as np
from decision_rules.classification.rule import ClassificationConclusion
from decision_rules.classification.rule import ClassificationRule
from decision_rules.core.coverage import Coverage
from tests.core.test_abstract_condition import TestingCondition


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


if __name__ == '__main__':
    unittest.main()
