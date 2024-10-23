# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

import numpy as np
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.ruleset_factories import ruleset_factory
from rulekit.classification import RuleClassifier

from tests.loaders import load_dataset_to_x_y


class TestComplementaryConditions(unittest.TestCase):

    rulekit_model: RuleClassifier
    dataset_path: str = 'classification/mushroom.csv'

    @classmethod
    def setUpClass(cls):
        X, y = load_dataset_to_x_y(cls.dataset_path)
        X = X.astype(str)
        y = y.astype(str)
        rulekit_model = RuleClassifier(
            complementary_conditions=True
        )
        rulekit_model.fit(X, y)
        cls.rulekit_model = rulekit_model

    def test_if_prediction_same_as_rulekit(self):
        X, y = load_dataset_to_x_y(TestComplementaryConditions.dataset_path)
        X = X.astype(str)
        y = y.astype(str)
        ruleset: ClassificationRuleSet = ruleset_factory(
            self.rulekit_model, X, y
        )

        y_pred: np.ndarray = ruleset.predict(X)
        y_pred_rulekit: np.ndarray = TestComplementaryConditions.rulekit_model.predict(
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
