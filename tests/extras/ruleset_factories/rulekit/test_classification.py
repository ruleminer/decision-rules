# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

import numpy as np
from decision_rules import measures
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.ruleset_factories import ruleset_factory
from decision_rules.serialization import JSONSerializer
from rulekit.classification import RuleClassifier
from rulekit.params import Measures

from tests.loaders import load_dataset_to_x_y


class TestClassificationRuleSet(unittest.TestCase):

    rulekit_model: RuleClassifier
    dataset_path: str = "classification/deals-train.csv"

    @classmethod
    def setUpClass(cls):
        X, y = load_dataset_to_x_y(cls.dataset_path)
        rulekit_model = RuleClassifier(
            induction_measure=Measures.C2,
            pruning_measure=Measures.C2,
            voting_measure=Measures.C2,
        )
        rulekit_model.fit(X, y)
        cls.rulekit_model = rulekit_model

    def test_if_prediction_same_as_rulekit(self):
        X, y = load_dataset_to_x_y(TestClassificationRuleSet.dataset_path)
        ruleset: ClassificationRuleSet = ruleset_factory(
            self.rulekit_model, X, y
        )

        y_pred: np.ndarray = ruleset.predict(X)
        y_pred_rulekit: np.ndarray = TestClassificationRuleSet.rulekit_model.predict(
            X)

        self.assertEqual(
            len(self.rulekit_model.model.rules),
            len(ruleset.rules),
            'RuleSet should contain the same number of rules as original RuleKit model'
        )

        self.assertTrue(
            np.array_equal(y_pred, y_pred_rulekit),
            'RuleSet should predict the same as original RuleKit model'
        )

    def test_serialization(self):
        X, y = load_dataset_to_x_y(TestClassificationRuleSet.dataset_path)
        ruleset: ClassificationRuleSet = ruleset_factory(
            self.rulekit_model, X, y
        )

        clf_serialized = JSONSerializer.serialize(ruleset)
        clf_deserialized: ClassificationRuleSet = JSONSerializer.deserialize(
            clf_serialized, ClassificationRuleSet
        )
        clf_deserialized.update(X, y, measure=measures.c2)

        y_pred: np.ndarray = self.rulekit_model.predict(X)
        y_pred_rulekit: np.ndarray = clf_deserialized.predict(X)
        self.assertTrue(
            np.array_equal(y_pred, y_pred_rulekit),
            'RuleSet should predict the same as original RuleKit model'
        )

    def test_if_subconditions_are_in_the_same_order(self):
        X, y = load_dataset_to_x_y(TestClassificationRuleSet.dataset_path)
        ruleset: ClassificationRuleSet = ruleset_factory(
            self.rulekit_model, X, y
        )

        for rule_index, rule in enumerate(ruleset.rules):
            for subcondition_index, subcondition in enumerate(rule.premise.subconditions):
                expected_condition_attr: str = ruleset.column_names[list(
                    subcondition.attributes)[0]]
                rulekit_subcondition = (
                    self.rulekit_model.model.rules[rule_index]._java_object
                    .getPremise()
                    .getSubconditions()
                    .get(subcondition_index)
                )
                actual_condition_attr: str = rulekit_subcondition.getAttributes().toArray()[
                    0]
                if expected_condition_attr != actual_condition_attr:
                    self.fail(
                        'Rules premised subconditions are expected to be in the same order as in original ruleset'
                    )
