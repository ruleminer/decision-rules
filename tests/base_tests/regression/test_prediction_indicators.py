# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os
import unittest

import numpy as np
import pandas as pd
from decision_rules import measures
from decision_rules.regression import RegressionConclusion
from decision_rules.regression import RegressionRuleSet
from decision_rules.regression.prediction_indicators import \
    calculate_for_regression
from tests.loaders import load_regression_dataset
from tests.loaders import load_regression_ruleset


class TestRegressionPredictionIndicators(unittest.TestCase):

    def setUp(self) -> None:
        df: pd.DataFrame = load_regression_dataset()
        self.X, self.y = df.drop('label', axis=1), df['label']
        self.ruleset: RegressionRuleSet = load_regression_ruleset()
        self.ruleset.update(self.X, self.y, measure=measures.c2)

    def test_prediction_indicators(self):
        y_pred: np.ndarray = self.ruleset.predict(self.X)
        indicators: dict = calculate_for_regression(
            y_true=self.y, y_pred=y_pred
        )
        self.assertTrue(isinstance(indicators, dict))

    def test_prediction_indicators_only_on_covered_examples(self):
        self.ruleset.rules = self.ruleset.rules[int(
            len(self.ruleset.rules) / 10 * 8):]
        self.ruleset.set_default_conclusion_enabled(False)
        y_pred: np.ndarray = self.ruleset.predict(self.X)

        with self.assertRaises(ValueError, msg='Indicators calculation will fail due to NaN values'):
            calculate_for_regression(
                y_true=self.y, y_pred=y_pred,
                calculate_only_for_covered_examples=False
            )
        indicators_on_covered_example = calculate_for_regression(
            y_true=self.y, y_pred=y_pred,
            calculate_only_for_covered_examples=True
        )
        self.assertGreater(
            43.0,
            indicators_on_covered_example['general']['MAE'],
        )
