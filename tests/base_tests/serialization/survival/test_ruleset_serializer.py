# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

import pandas as pd

from decision_rules.conditions import (AttributesCondition, CompoundCondition,
                                       ElementaryCondition, LogicOperators,
                                       NominalCondition)
from decision_rules.core.coverage import Coverage
from decision_rules.core.exceptions import InvalidStateError
from decision_rules.serialization import JSONSerializer, SerializationModes
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator
from decision_rules.survival.rule import SurvivalConclusion, SurvivalRule
from decision_rules.survival.ruleset import SurvivalRuleSet
from tests.helpers import compare_survival_prediction


class TestSurvivalRuleSetSerializer(unittest.TestCase):

    def _prepare_ruleset(self) -> SurvivalRuleSet:
        rule1 = SurvivalRule(
            CompoundCondition(
                subconditions=[
                    AttributesCondition(column_left=2, column_right=3, operator=">"),
                    ElementaryCondition(
                        column_index=2,
                        left=-1,
                        right=2.0,
                        left_closed=True,
                        right_closed=False,
                    ),
                    NominalCondition(
                        column_index=2,
                        value="value",
                    ),
                ],
                logic_operator=LogicOperators.ALTERNATIVE,
            ),
            conclusion=SurvivalConclusion(1.0, column_name="label"),
            column_names=["col_1", "col_2", "col_3", "col_4", "survival_time"],
            survival_time_attr="survival_time",
        )
        rule1.coverage = Coverage(p=10, n=2, P=12, N=20)
        rule1.conclusion.median_survival_time_ci_lower = 1.0
        rule1.conclusion.median_survival_time_ci_upper = 3.0
        rule2 = SurvivalRule(
            CompoundCondition(
                subconditions=[
                    AttributesCondition(column_left=1, column_right=3, operator="="),
                    ElementaryCondition(
                        column_index=2,
                        left=float("-inf"),
                        right=3.0,
                        left_closed=False,
                        right_closed=False,
                    ),
                ],
                logic_operator=LogicOperators.CONJUNCTION,
            ),
            conclusion=SurvivalConclusion(1.0, column_name="label"),
            column_names=["col_1", "col_2", "col_3", "col_4", "survival_time"],
            survival_time_attr="survival_time",
        )
        rule2.coverage = Coverage(p=19, n=1, P=20, N=12)
        rule2.conclusion.median_survival_time_ci_lower = 1.0
        rule2.conclusion.median_survival_time_ci_upper = 3.0
        ruleset = SurvivalRuleSet(  # pylint: disable=abstract-class-instantiated
            rules=[rule1, rule2], survival_time_attr="survival_time"
        )
        conclusion_estimator_dict = {
            "times": [1.1, 2.1, 3.0, 10, 22.2],
            "events_count": [1.0, 0, 1.0, 1.0, 0],
            "censored_count": [0.0, 1.0, 2.0, 0.0, 5.0],
            "at_risk_count": [75.0, 21.0, 5.0, 1.0, 0.0],
            "probabilities": [0.98, 0.92, 0.76, 0.52, 0.31],
        }
        ruleset.default_conclusion = SurvivalConclusion(
            value=0.0, column_name=ruleset.rules[0].conclusion.column_name
        )
        ruleset.default_conclusion.estimator = KaplanMeierEstimator().update(
            conclusion_estimator_dict, update_additional_indicators=True
        )
        ruleset.column_names = ["col_1", "col_2", "col_3", "col_4", "survival_time"]
        return ruleset

    def _prepare_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        X: pd.DataFrame = pd.DataFrame(
            {
                "col_1": range(5),
                "col_2": range(5),
                "col_3": range(5),
                "col_4": range(5),
                "survival_time": range(5),
            }
        )
        y: pd.Series = pd.Series(["0", "1", "0", "1", "0"])
        return X, y

    def test_serializing_deserializing(self):
        ruleset: SurvivalRuleSet = self._prepare_ruleset()
        ruleset.update(*self._prepare_dataset())
        serialized_ruleset = JSONSerializer.serialize(ruleset)
        deserializer_ruleset: SurvivalRuleSet = JSONSerializer.deserialize(
            serialized_ruleset, SurvivalRuleSet
        )

        self.assertEqual(
            ruleset,
            deserializer_ruleset,
            "Serializing and deserializing should lead to the the same object",
        )

    def test_prediction_after_deserializing_without_update(self):
        ruleset: SurvivalRuleSet = self._prepare_ruleset()
        ruleset.update(*self._prepare_dataset())
        X, y = self._prepare_dataset()
        ruleset.update(X, y)
        # change conclusion value so it will be different than the default one
        ruleset.default_conclusion.value += 1

        serialized_ruleset_min: dict = JSONSerializer.serialize(
            ruleset, mode=SerializationModes.MINIMAL
        )
        serialized_ruleset_full: dict = JSONSerializer.serialize(
            ruleset, mode=SerializationModes.FULL
        )
        deserializer_ruleset_min: SurvivalRuleSet = JSONSerializer.deserialize(
            serialized_ruleset_min, SurvivalRuleSet
        )
        deserialized_ruleset_full: SurvivalRuleSet = JSONSerializer.deserialize(
            serialized_ruleset_full, SurvivalRuleSet
        )
        with self.assertRaises(InvalidStateError):
            deserializer_ruleset_min.predict(X)

        self.assertEqual(
            ruleset.default_conclusion.value,
            deserialized_ruleset_full.default_conclusion.value,
            "Default conclusion after deserializing should be the same",
        )
        self.assertEqual(
            ruleset.default_conclusion.value,
            deserializer_ruleset_min.default_conclusion.value,
            "Default conclusion after deserializing should be the same",
        )
        self.assertEqual(
            [r.coverage for r in ruleset.rules],
            [r.coverage for r in deserialized_ruleset_full.rules],
            "Coverages after deserializing should be the same",
        )
        self.assertEqual(
            [r.voting_weight for r in ruleset.rules],
            [r.voting_weight for r in deserialized_ruleset_full.rules],
            "Voting weights after deserializing should be the same",
        )
        self.assertTrue(
            compare_survival_prediction(
                deserialized_ruleset_full.predict(X).tolist(),
                ruleset.predict(X).tolist(),
            ),
            "Prediction after deserializing should be the same",
        )


if __name__ == "__main__":
    unittest.main()
