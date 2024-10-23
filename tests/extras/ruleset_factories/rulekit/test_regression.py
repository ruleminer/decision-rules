# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

import numpy as np
import pandas as pd
from decision_rules import measures
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.ruleset_factories import ruleset_factory
from decision_rules.serialization import JSONSerializer
from rulekit.params import Measures
from rulekit.regression import RuleRegressor

from tests.loaders import load_dataset_to_x_y


class TestRegressionRuleSet(unittest.TestCase):

    rulekit_model: RuleRegressor
    dataset_path = "regression/methane-train-minimal.csv"

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        use_mean_based_regression: bool
    ) -> RuleRegressor:
        X, y = load_dataset_to_x_y(self.dataset_path)
        rulekit_model = RuleRegressor(
            voting_measure=Measures.C2,
            mean_based_regression=use_mean_based_regression
        )
        rulekit_model.fit(X, y)
        return rulekit_model

    def test_if_prediction_same_as_rulekit(self):
        X, y = load_dataset_to_x_y(self.dataset_path)
        reg_median = self.train_model(X, y, use_mean_based_regression=False)
        reg_mean = self.train_model(X, y, use_mean_based_regression=True)

        with self.assertRaises(NotImplementedError):
            ruleset_factory(reg_median, X, y)

        ruleset_mean: RegressionRuleSet = ruleset_factory(
            reg_mean, X, y
        )
        y_pred: np.ndarray = ruleset_mean.predict(X)
        y_pred_rulekit: np.ndarray = reg_mean.predict(X)

        self.assertEqual(
            len(reg_mean.model.rules),
            len(ruleset_mean.rules),
            'RuleSet should contain the same number of rules as original RuleKit model'
        )
        self.assertTrue(
            np.allclose(y_pred, y_pred_rulekit, atol=1.0e-10),
            'RuleSet should predict the same as original RuleKit model'
        )

    def test_serialization(self):
        X, y = load_dataset_to_x_y(self.dataset_path)
        reg = self.train_model(X, y, use_mean_based_regression=True)

        ruleset_mean: RegressionRuleSet = ruleset_factory(
            reg, X, y
        )

        reg_serialized = JSONSerializer.serialize(ruleset_mean)
        reg_deserialized: RegressionRuleSet = JSONSerializer.deserialize(
            reg_serialized, RegressionRuleSet
        )
        reg_deserialized.update(X, y, measure=measures.c2)

        y_pred: np.ndarray = ruleset_mean.predict(X)
        y_pred_rulekit: np.ndarray = reg_deserialized.predict(X)
        self.assertTrue(
            np.allclose(y_pred, y_pred_rulekit, atol=0.00001),
            'RuleSet should predict the same as original RuleKit model'
        )
