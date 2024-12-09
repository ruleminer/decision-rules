# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

from decision_rules.conditions import AttributesCondition
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import ElementaryCondition
from decision_rules.conditions import LogicOperators
from decision_rules.conditions import NominalCondition
from decision_rules.core.coverage import Coverage
from decision_rules.serialization import JSONSerializer
from decision_rules.survival.rule import SurvivalConclusion
from decision_rules.survival.rule import SurvivalRule


class TestSurvivalRuleSerializer(unittest.TestCase):

    def test_serializing_deserializing(self):
        rule = SurvivalRule(
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
            conclusion=SurvivalConclusion(
                value=2.0,
                column_name='class'
            ),
            column_names=['col_1', 'col_2', 'col_3', 'col_4', 'survival_time'],
            survival_time_attr='survival_time'
        )
        rule.coverage = Coverage(p=10, n=2, P=12, N=20)
        rule.conclusion.median_survival_time_ci_lower = 1.0
        rule.conclusion.median_survival_time_ci_upper = 3.0

        serialized_rule = JSONSerializer.serialize(rule)
        deserializer_rule: SurvivalRule = JSONSerializer.deserialize(
            serialized_rule,
            SurvivalRule
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
