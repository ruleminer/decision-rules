# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

import pandas as pd
from decision_rules import measures
from decision_rules.conditions import AttributesCondition
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import ElementaryCondition
from decision_rules.conditions import LogicOperators
from decision_rules.conditions import NominalCondition
from decision_rules.core.coverage import Coverage
from decision_rules.regression.rule import RegressionConclusion
from decision_rules.regression.rule import RegressionRule
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.serialization import JSONSerializer


class TestRegressionRuleSetSerializer(unittest.TestCase):

    def test_serializing_deserializing(self):
        rule1 = RegressionRule(
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
                1.0, low=0.5, high=1.5, column_name='label'),
            column_names=['col_1', 'col_2', 'col_3', 'col_4']
        )
        rule1.coverage = Coverage(p=10, n=2, P=12, N=20)
        rule2 = RegressionRule(
            CompoundCondition(
                subconditions=[
                    AttributesCondition(
                        column_left=1, column_right=3, operator='='
                    ),
                    ElementaryCondition(
                        column_index=2,
                        left=float('-inf'),
                        right=3.0,
                        left_closed=False,
                        right_closed=False
                    ),
                ],
                logic_operator=LogicOperators.CONJUNCTION
            ),
            conclusion=RegressionConclusion(
                1.0, low=0.5, high=1.5, column_name='label'),
            column_names=['col_1', 'col_2', 'col_3', 'col_4']
        )
        rule2.coverage = Coverage(p=19, n=1, P=20, N=12)
        ruleset = RegressionRuleSet([rule1, rule2])
        X: pd.DataFrame = pd.DataFrame({
            'col_1': range(6),
            'col_2': range(6),
            'col_3': range(6),
            'col_4': range(6),
        })
        y: pd.Series = pd.Series([1, 2, 2, 1, 2, 1])
        ruleset.update(X, y, measure=measures.accuracy)

        serialized_ruleset = JSONSerializer.serialize(ruleset)
        deserializer_ruleset = JSONSerializer.deserialize(
            serialized_ruleset, RegressionRuleSet
        )

        self.assertEqual(
            ruleset, deserializer_ruleset,
            'Serializing and deserializing should lead to the the same object'
        )


if __name__ == '__main__':
    unittest.main()
