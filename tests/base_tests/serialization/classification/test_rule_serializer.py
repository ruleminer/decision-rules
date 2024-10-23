# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

from decision_rules.classification.rule import ClassificationConclusion
from decision_rules.classification.rule import ClassificationRule
from decision_rules.conditions import AttributesCondition
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import ElementaryCondition
from decision_rules.conditions import LogicOperators
from decision_rules.conditions import NominalCondition
from decision_rules.core.coverage import Coverage
from decision_rules.serialization import JSONSerializer


class TestClassificationRuleSerializer(unittest.TestCase):

    def test_serializing_deserializing(self):
        rule = ClassificationRule(
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
            conclusion=ClassificationConclusion(
                value=2,
                column_name='class'
            ),
            column_names=list(range(4))
        )
        rule.coverage = Coverage(p=10, n=2, P=12, N=20)

        serialized_rule = JSONSerializer.serialize(rule)
        deserializer_rule: ClassificationRule = JSONSerializer.deserialize(
            serialized_rule,
            ClassificationRule
        )
        # column_names cannot be populated while deserializing rule without ruleset
        rule.column_names = []
        # P and N cannot be populated while deserializing rule without ruleset
        rule.coverage.P = None
        rule.coverage.N = None
        deserializer_rule.conclusion.column_name = 'class'
        self.assertEqual(
            rule, deserializer_rule,
            'Serializing and deserializing should lead to the the same object'
        )

    def test_serializing_deserializing_without_coverage(self):
        rule = ClassificationRule(
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
            conclusion=ClassificationConclusion(
                value=2,
                column_name='class'
            ),
            column_names=list(range(4))
        )

        serialized_rule = JSONSerializer.serialize(rule)
        deserializer_rule: ClassificationRule = JSONSerializer.deserialize(
            serialized_rule,
            ClassificationRule
        )
        # column_names cannot be populated while deserializing rule without ruleset
        rule.column_names = []
        deserializer_rule.conclusion.column_name = 'class'
        self.assertEqual(
            rule, deserializer_rule,
            'Serializing and deserializing should lead to the the same object'
        )


if __name__ == '__main__':
    unittest.main()
