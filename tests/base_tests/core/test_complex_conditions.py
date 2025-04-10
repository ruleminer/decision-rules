# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

import numpy as np

from decision_rules.conditions import (AttributesRelationCondition,
                                       DiscreteSetCondition,
                                       NominalAttributesEqualityCondition)


class TestAttributesCondition(unittest.TestCase):

    def test_covered_mask(self):
        X = np.array(
            [
                [1, 2],
                [1, 0],
                [1, 1],
            ]
        )
        cond = AttributesRelationCondition(column_left=0, column_right=1, operator="=")
        self.assertTrue(
            np.array_equal(cond.covered_mask(X), np.array([False, False, True])),
            "Should work for = operator",
        )
        cond = AttributesRelationCondition(column_left=0, column_right=1, operator="!=")
        self.assertTrue(
            np.array_equal(cond.covered_mask(X), np.array([True, True, False])),
            "Should work for != operator",
        )

        cond = AttributesRelationCondition(column_left=0, column_right=1, operator=">")
        self.assertTrue(
            np.array_equal(cond.covered_mask(X), np.array([False, True, False])),
            "Should work for > operator",
        )

        cond = AttributesRelationCondition(column_left=0, column_right=1, operator=">=")
        self.assertTrue(
            np.array_equal(cond.covered_mask(X), np.array([False, True, True])),
            "Should work for >= operator",
        )

        cond = AttributesRelationCondition(column_left=0, column_right=1, operator="<")
        self.assertTrue(
            np.array_equal(cond.covered_mask(X), np.array([True, False, False])),
            "Should work for < operator",
        )

        cond = AttributesRelationCondition(column_left=0, column_right=1, operator="<=")
        self.assertTrue(
            np.array_equal(cond.covered_mask(X), np.array([True, False, True])),
            "Should work for <= operator",
        )

        with self.assertRaises(ValueError):
            cond = AttributesRelationCondition(
                column_left=0, column_right=1, operator="invalid"
            )

    def test_equality(self):
        cond_1 = AttributesRelationCondition(column_left=0, column_right=1, operator="=")
        cond_2 = AttributesRelationCondition(column_left=0, column_right=1, operator="=")
        self.assertTrue(cond_1 == cond_2)

        cond_2.negated = True
        self.assertTrue(cond_1 != cond_2)

        cond_1.operator = "!="
        self.assertTrue(cond_1 != cond_2)
        cond_1.operator = "="

        cond_1.column_left = 2
        self.assertTrue(cond_1 != cond_2)


class TestNominalAttributesEqualityCondition(unittest.TestCase):

    def test_covered_mask(self):
        X = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ]
        )
        cond = NominalAttributesEqualityCondition(column_indices=[0, 1])
        self.assertTrue(
            np.array_equal(
                cond.covered_mask(X), np.array([True, False, True, False, True])
            ),
            "Should work for more than two columns",
        )

        cond = NominalAttributesEqualityCondition(column_indices=[0, 1, 2])
        self.assertTrue(
            np.array_equal(
                cond.covered_mask(X), np.array([True, False, True, False, False])
            ),
            "Should work for more than two columns",
        )

    def test_equality(self):
        cond_1 = NominalAttributesEqualityCondition(column_indices=[0, 1])
        cond_2 = NominalAttributesEqualityCondition(column_indices=[0, 1])
        self.assertTrue(cond_1 == cond_2)

        cond_1.column_indices.append(2)
        self.assertTrue(cond_1 != cond_2)
        cond_1.column_indices = [0, 1]

        cond_2.negated = True
        self.assertTrue(cond_1 != cond_2)


class DiscreteSetConditionCondition(unittest.TestCase):

    def test_covered_mask(self):
        X = np.array(
            [
                [0],
                [1],
                [2],
                [3],
                [4],
            ]
        )
        cond = DiscreteSetCondition(column_index=0, values_set={0, 1, 2})
        self.assertTrue(
            np.array_equal(
                cond.covered_mask(X), np.array([True, True, True, False, False])
            ),
        )

    def test_equality(self):
        cond_1 = DiscreteSetCondition(column_index=0, values_set={0, 1, 2})
        cond_2 = DiscreteSetCondition(column_index=0, values_set={0, 1, 2})
        self.assertTrue(cond_1 == cond_2)

        cond_1.values_set = {0, 1}
        self.assertTrue(cond_1 != cond_2)


if __name__ == "__main__":
    unittest.main()
