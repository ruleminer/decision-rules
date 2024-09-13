# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

import numpy as np
from decision_rules.conditions import NominalCondition


class TestNominalCondition(unittest.TestCase):

    def setUp(self):
        pass

    def test_covered_mask(self):
        X: np.ndarray = np.array(np.random.randint(0, 2, size=10))
        X = np.expand_dims(X, axis=1)
        condition = NominalCondition(
            column_index=0, value=1
        )

        expected_mask: np.ndarray = X[:, 0].astype(bool)
        actual_mask: np.ndarray = condition.covered_mask(X)
        self.assertTrue(
            np.array_equal(expected_mask, actual_mask),
            'Covered examples mask does not work properly'
        )

        condition.negated = True
        expected_mask: np.ndarray = np.logical_not(X[:, 0].astype(bool))
        actual_mask: np.ndarray = condition.covered_mask(X)
        self.assertTrue(
            np.array_equal(expected_mask, actual_mask),
            'Covered examples mask does not work properly for negated condition'
        )

    def test_equality(self):
        cond_1 = NominalCondition(column_index=0, value=1)
        cond_2 = NominalCondition(column_index=0, value=1)
        self.assertTrue(cond_1 == cond_2)

        cond_2.negated = True
        self.assertTrue(cond_1 != cond_2)

        cond_1 = NominalCondition(column_index=0, value=1)
        cond_2 = NominalCondition(column_index=1, value=1)
        self.assertTrue(cond_1 != cond_2)

        cond_1 = NominalCondition(column_index=0, value=1)
        cond_2 = NominalCondition(column_index=0, value=2)
        self.assertTrue(cond_1 != cond_2)


if __name__ == '__main__':
    unittest.main()
