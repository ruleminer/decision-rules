# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

import pandas as pd

from decision_rules import measures
from decision_rules.conditions import (AttributesCondition, CompoundCondition,
                                       ElementaryCondition, LogicOperators,
                                       NominalCondition)
from decision_rules.core.coverage import Coverage
from decision_rules.regression.rule import RegressionConclusion, RegressionRule
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.serialization import JSONSerializer


class TestRegressionRuleSetSerializer(unittest.TestCase):

    def _prepare_ruleset(self) -> RegressionRuleSet:
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
        return RegressionRuleSet([rule1, rule2]) # pylint: disable=abstract-class-instantiated

    def _prepare_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        X: pd.DataFrame = pd.DataFrame({
            'col_1': range(6),
            'col_2': range(6),
            'col_3': range(6),
            'col_4': range(6),
        })
        y: pd.Series = pd.Series([1, 2, 2, 1, 2, 1])
        return X, y

    def test_serializing_deserializing(self):
        ruleset: RegressionRuleSet = self._prepare_ruleset()
        X, y = self._prepare_dataset()
        ruleset.update(X, y, measure=measures.accuracy)

        serialized_ruleset = JSONSerializer.serialize(ruleset)
        deserializer_ruleset = JSONSerializer.deserialize(
            serialized_ruleset, RegressionRuleSet
        )

        self.assertEqual(
            ruleset, deserializer_ruleset,
            'Serializing and deserializing should lead to the the same object'
        )

    def test_prediction_after_deserializing_without_update(self):
        ruleset: RegressionRuleSet = self._prepare_ruleset()
        X, y = self._prepare_dataset()
        ruleset.update(X, y, measure=measures.accuracy)
        # change conclusion value so it will be different than the default one
        ruleset.default_conclusion.high += 0.01 
        ruleset.default_conclusion.value += 0.01 

        serialized_ruleset: dict = JSONSerializer.serialize(ruleset)
        deserializer_ruleset: RegressionRuleSet = JSONSerializer.deserialize(
            serialized_ruleset, RegressionRuleSet
        )

        self.assertEqual(
            ruleset.default_conclusion.value,
            deserializer_ruleset.default_conclusion.value,
            'Default conclusion after deserializing should be the same'
        )
        self.assertEqual(
            [r.coverage for r in ruleset.rules],
            [r.coverage for r in deserializer_ruleset.rules],
            'Coverages after deserializing should be the same'
        )
        self.assertEqual(
            [r.voting_weight for r in ruleset.rules],
            [r.voting_weight for r in deserializer_ruleset.rules],
            'Voting weights after deserializing should be the same'
        )
        self.assertEqual(
            ruleset.predict(X).tolist(),
            deserializer_ruleset.predict(X).tolist(),
            'Prediction after deserializing should be the same'
        )


if __name__ == '__main__':
    unittest.main()
