# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

import numpy as np

from decision_rules.conditions import AttributesRelationCondition
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import DiscreteSetCondition
from decision_rules.conditions import LogicOperators
from decision_rules.conditions import NominalAttributesEqualityCondition
from decision_rules.conditions import NominalCondition
from decision_rules.conditions import count_leaf_conditions


class TestAttributesCondition(unittest.TestCase):

    def test_covered_mask(self):
        X = np.array(
            [
                [1, 2],
                [1, 0],
                [1, 1],
            ]
        )
        cond = AttributesRelationCondition(
            column_left=0, column_right=1, operator="=")
        self.assertTrue(
            np.array_equal(cond.covered_mask(
                X), np.array([False, False, True])),
            "Should work for = operator",
        )
        cond = AttributesRelationCondition(
            column_left=0, column_right=1, operator="!=")
        self.assertTrue(
            np.array_equal(cond.covered_mask(
                X), np.array([True, True, False])),
            "Should work for != operator",
        )

        cond = AttributesRelationCondition(
            column_left=0, column_right=1, operator=">")
        self.assertTrue(
            np.array_equal(cond.covered_mask(
                X), np.array([False, True, False])),
            "Should work for > operator",
        )

        cond = AttributesRelationCondition(
            column_left=0, column_right=1, operator=">=")
        self.assertTrue(
            np.array_equal(cond.covered_mask(
                X), np.array([False, True, True])),
            "Should work for >= operator",
        )

        cond = AttributesRelationCondition(
            column_left=0, column_right=1, operator="<")
        self.assertTrue(
            np.array_equal(cond.covered_mask(
                X), np.array([True, False, False])),
            "Should work for < operator",
        )

        cond = AttributesRelationCondition(
            column_left=0, column_right=1, operator="<=")
        self.assertTrue(
            np.array_equal(cond.covered_mask(
                X), np.array([True, False, True])),
            "Should work for <= operator",
        )

        with self.assertRaises(ValueError):
            cond = AttributesRelationCondition(
                column_left=0, column_right=1, operator="invalid"
            )

    def test_equality(self):
        cond_1 = AttributesRelationCondition(
            column_left=0, column_right=1, operator="=")
        cond_2 = AttributesRelationCondition(
            column_left=0, column_right=1, operator="=")
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
                cond.covered_mask(X), np.array(
                    [True, False, True, False, True])
            ),
            "Should work for more than two columns",
        )

        cond = NominalAttributesEqualityCondition(column_indices=[0, 1, 2])
        self.assertTrue(
            np.array_equal(
                cond.covered_mask(X), np.array(
                    [True, False, True, False, False])
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
                cond.covered_mask(X), np.array(
                    [True, True, True, False, False])
            ),
        )

    def test_equality(self):
        cond_1 = DiscreteSetCondition(column_index=0, values_set={0, 1, 2})
        cond_2 = DiscreteSetCondition(column_index=0, values_set={0, 1, 2})
        self.assertTrue(cond_1 == cond_2)

        cond_1.values_set = {0, 1}
        self.assertTrue(cond_1 != cond_2)


class TestCompoundCondition(unittest.TestCase):

    def setUp(self):
        self.X = np.array(
            [
                ["0", "0"],
                ["0", "1"],
                ["1", "0"],
                ["1", "1"],
            ],
            dtype=object,
        )
        self.a_eq_0 = NominalCondition(column_index=0, value="0")
        self.b_eq_0 = NominalCondition(column_index=1, value="0")

    def test_conjunction_covered_mask(self):
        cond = CompoundCondition(
            subconditions=[self.a_eq_0, self.b_eq_0],
            logic_operator=LogicOperators.CONJUNCTION,
        )
        self.assertTrue(
            np.array_equal(
                cond.covered_mask(self.X), np.array([True, False, False, False])
            ),
            "Conjunction should cover rows matching all subconditions",
        )

    def test_alternative_covered_mask(self):
        cond = CompoundCondition(
            subconditions=[self.a_eq_0, self.b_eq_0],
            logic_operator=LogicOperators.ALTERNATIVE,
        )
        self.assertTrue(
            np.array_equal(
                cond.covered_mask(self.X), np.array([True, True, True, False])
            ),
            "Alternative should cover rows matching any subcondition",
        )

    def test_empty_conjunction_covered_mask(self):
        cond = CompoundCondition(
            subconditions=[],
            logic_operator=LogicOperators.CONJUNCTION,
        )
        self.assertTrue(
            np.array_equal(
                cond.covered_mask(self.X), np.array([False, False, False, False])
            ),
            "Empty conjunction should cover no rows",
        )

    def test_empty_alternative_covered_mask(self):
        cond = CompoundCondition(
            subconditions=[],
            logic_operator=LogicOperators.ALTERNATIVE,
        )
        self.assertTrue(
            np.array_equal(
                cond.covered_mask(self.X), np.array([True, True, True, True])
            ),
            "Empty alternative should cover all rows",
        )

    def test_ghost_conjunction_with_empty_alternative(self):
        cond = CompoundCondition(
            subconditions=[
                CompoundCondition(
                    subconditions=[],
                    logic_operator=LogicOperators.ALTERNATIVE,
                )
            ],
            logic_operator=LogicOperators.CONJUNCTION,
        )
        self.assertTrue(
            np.array_equal(
                cond.covered_mask(self.X), np.array([False, False, False, False])
            ),
            "Conjunction with no leaf conditions should cover no rows",
        )

    def test_count_leaf_conditions(self):
        self.assertEqual(
            count_leaf_conditions(
                CompoundCondition(
                    subconditions=[self.a_eq_0, self.b_eq_0],
                    logic_operator=LogicOperators.CONJUNCTION,
                )
            ),
            2,
            "Should count leaf subconditions in a compound condition",
        )
        self.assertEqual(
            count_leaf_conditions(
                CompoundCondition(
                    subconditions=[],
                    logic_operator=LogicOperators.CONJUNCTION,
                )
            ),
            0,
            "Empty compound condition should have zero leaf conditions",
        )
        self.assertEqual(
            count_leaf_conditions(
                CompoundCondition(
                    subconditions=[
                        CompoundCondition(
                            subconditions=[],
                            logic_operator=LogicOperators.ALTERNATIVE,
                        )
                    ],
                    logic_operator=LogicOperators.CONJUNCTION,
                )
            ),
            0,
            "Ghost branches without leaf conditions should count as zero",
        )


if __name__ == "__main__":
    unittest.main()
