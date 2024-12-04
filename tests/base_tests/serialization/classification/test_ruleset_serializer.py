# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

import pandas as pd

from decision_rules import measures
from decision_rules.classification.rule import (ClassificationConclusion,
                                                ClassificationRule)
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.conditions import (AttributesCondition, CompoundCondition,
                                       ElementaryCondition, LogicOperators,
                                       NominalCondition)
from decision_rules.core.coverage import Coverage
from decision_rules.core.exceptions import InvalidStateError
from decision_rules.serialization import JSONSerializer, SerializationModes


class TestClassificationRuleSetSerializer(unittest.TestCase):

    def _prepare_ruleset(self) -> ClassificationRuleSet:
        rule1 = ClassificationRule(
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
            conclusion=ClassificationConclusion(2, column_name="class"),
            column_names=list(range(4)),
        )
        rule1.coverage = Coverage(p=10, n=2, P=12, N=20)
        rule2 = ClassificationRule(
            CompoundCondition(
                subconditions=[
                    AttributesCondition(column_left=1, column_right=3, operator="<"),
                    NominalCondition(
                        column_index=1,
                        value="value",
                    ),
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
            conclusion=ClassificationConclusion(1, column_name="class"),
            column_names=list(range(4)),
        )
        rule2.coverage = Coverage(p=19, n=1, P=20, N=12)
        return ClassificationRuleSet(
            [rule1, rule2]
        )  # pylint: disable=abstract-class-instantiated

    def _prepare_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        X: pd.DataFrame = pd.DataFrame(
            {
                "col_1": range(6),
                "col_2": range(6),
                "col_3": range(6),
                "col_4": range(6),
            }
        )
        y: pd.Series = pd.Series([1, 2, 2, 1, 2, 1])
        return X, y

    def test_serializing_deserializing_with_normal_update(self):
        ruleset: ClassificationRuleSet = self._prepare_ruleset()
        X, y = self._prepare_dataset()

        ruleset.update(X, y, measure=measures.accuracy)

        serialized_ruleset = JSONSerializer.serialize(ruleset)
        deserializer_ruleset = JSONSerializer.deserialize(
            serialized_ruleset, ClassificationRuleSet
        )
        ruleset.predict(X)

        self.assertEqual(
            ruleset,
            deserializer_ruleset,
            "Serializing and deserializing should lead to the the same object",
        )

    def test_serializing_deserializing_using_update_using_coverages(self):
        ruleset: ClassificationRuleSet = self._prepare_ruleset()
        X, _ = self._prepare_dataset()
        ruleset.update_using_coverages(
            coverages_info={
                rule.uuid: {"p": 10, "n": 0, "P": 20, "N": 20} for rule in ruleset.rules
            },
            measure=measures.accuracy,
            columns_names=X.columns.tolist(),
        )
        ruleset.predict(X)

        serialized_ruleset = JSONSerializer.serialize(ruleset)
        deserializer_ruleset = JSONSerializer.deserialize(
            serialized_ruleset, ClassificationRuleSet
        )

        self.assertEqual(
            ruleset,
            deserializer_ruleset,
            "Serializing and deserializing should lead to the the same object",
        )

    def test_serializing_deserializing_without_coverages(self):
        ruleset: ClassificationRuleSet = self._prepare_ruleset()
        X, y = self._prepare_dataset()

        ruleset.update(X, y, measure=measures.accuracy)

        serialized_ruleset = JSONSerializer.serialize(ruleset)
        deserializer_ruleset = JSONSerializer.deserialize(
            serialized_ruleset, ClassificationRuleSet
        )
        ruleset.predict(X)

        self.assertEqual(
            ruleset,
            deserializer_ruleset,
            "Serializing and deserializing should lead to the the same object",
        )

    def test_prediction_after_deserializing_without_update(self):
        ruleset: ClassificationRuleSet = self._prepare_ruleset()
        X, y = self._prepare_dataset()
        ruleset.update(X, y, measure=measures.accuracy)
        # change conclusion value so it will be different than the default one
        ruleset.default_conclusion.value += 1

        serialized_ruleset_min: dict = JSONSerializer.serialize(
            ruleset, mode=SerializationModes.MINIMAL
        )
        serialized_ruleset_full: dict = JSONSerializer.serialize(
            ruleset, mode=SerializationModes.FULL
        )
        deserializer_ruleset_min: ClassificationRuleSet = JSONSerializer.deserialize(
            serialized_ruleset_min, ClassificationRuleSet
        )
        deserialized_ruleset_full: ClassificationRuleSet = JSONSerializer.deserialize(
            serialized_ruleset_full, ClassificationRuleSet
        )
        with self.assertRaises(InvalidStateError):
            deserializer_ruleset_min.predict(X)

        self.assertEqual(
            ruleset.default_conclusion.value,
            deserialized_ruleset_full.default_conclusion.value,
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
        self.assertEqual(
            ruleset.predict(X).tolist(),
            deserialized_ruleset_full.predict(X).tolist(),
            "Prediction after deserializing should be the same",
        )


if __name__ == "__main__":
    unittest.main()
