# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

from decision_rules.conditions import AttributesCondition
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import ElementaryCondition
from decision_rules.conditions import LogicOperators
from decision_rules.conditions import NominalCondition
from decision_rules.core.coverage import Coverage
from decision_rules.regression.rule import RegressionConclusion
from decision_rules.regression.rule import RegressionRule
from decision_rules.serialization import JSONSerializer


class TestRegressionRuleSerializer(unittest.TestCase):

    def test_serializing_deserializing(self):
        rule = RegressionRule(
            CompoundCondition(
                subconditions=[
                    AttributesCondition(
                        column_left=2, column_right=3, operator='>'
                    ),
                    ElementaryCondition(
                        column_index=2, left=-1, right=2.0, left_closed=True, right_closed=False
                    ),
                    NominalCondition(
                        column_index=2,
                        value='value',
                    )
                ],
                logic_operator=LogicOperators.ALTERNATIVE
            ),
            conclusion=RegressionConclusion(
                value=2.0,
                low=1.0,
                high=3.0,
                column_name='class'
            ),
            column_names=list(range(4))
        )
        rule.coverage = Coverage(p=10, n=2, P=12, N=20)
        rule.conclusion.train_covered_y_mean = 1.0
        rule.conclusion.train_covered_y_std = 0.13

        serialized_rule = JSONSerializer.serialize(rule)
        deserializer_rule: RegressionRule = JSONSerializer.deserialize(
            serialized_rule,
            RegressionRule
        )
        # column_names cannot be populated while deserializing rule without ruleset
        rule.column_names = []
        # P and N cannot be populated while deserializing rule without ruleset
        rule.coverage.P = None
        rule.coverage.N = None
        deserializer_rule.conclusion.column_name = rule.conclusion.column_name

        self.assertEqual(
            rule, deserializer_rule,
            'Serializing and deserializing should lead to the the same object'
        )


if __name__ == '__main__':
    unittest.main()
