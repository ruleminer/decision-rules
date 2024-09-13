# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os
import unittest

import numpy as np
import pandas as pd
from decision_rules.survival import SurvivalConclusion
from decision_rules.survival import SurvivalRuleSet
from decision_rules.survival.prediction_indicators import \
    calculate_for_survival
from tests.loaders import load_survival_dataset
from tests.loaders import load_survival_ruleset

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestSurvivalPredictionIndicators(unittest.TestCase):

    def setUp(self) -> None:
        df: pd.DataFrame = load_survival_dataset()
        self.X, self.y = (
            df.drop('survival_status', axis=1),
            df['survival_status']
        )
        self.ruleset: SurvivalRuleSet = load_survival_ruleset()
        self.ruleset.update(self.X, self.y)

    def test_prediction_indicators(self):
        y_pred: np.ndarray = self.ruleset.predict(self.X)
        indicators: dict = calculate_for_survival(
            self.ruleset, self.X, self.y, y_pred
        )
        self.assertTrue(isinstance(indicators, dict))

    def test_prediction_indicators_only_on_covered_examples(self):
        self.ruleset.rules = self.ruleset.rules[int(
            len(self.ruleset.rules) / 10 * 8):]
        self.ruleset.set_default_conclusion_enabled(False)
        y_pred: np.ndarray = self.ruleset.predict(self.X)

        indicators_on_full_dataset = calculate_for_survival(
            self.ruleset, self.X, self.y, y_pred,
            calculate_only_for_covered_examples=False
        )
        indicators_on_covered_example = calculate_for_survival(
            self.ruleset, self.X, self.y, y_pred,
            calculate_only_for_covered_examples=True
        )
        self.assertGreater(
            indicators_on_covered_example['general']['ibs'],
            indicators_on_full_dataset['general']['ibs'],
        )
