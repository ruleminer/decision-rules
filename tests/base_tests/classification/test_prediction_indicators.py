# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os
import unittest

import numpy as np
import pandas as pd
from decision_rules import measures
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.classification.prediction_indicators import \
    calculate_for_classification
from tests.loaders import load_classification_dataset
from tests.loaders import load_classification_ruleset
from tests.loaders import load_resources_path


class TestClassificationPredictionIndicators(unittest.TestCase):

    def setUp(self) -> None:
        df: pd.DataFrame = load_classification_dataset()
        self.X, self.y = df.drop('Salary', axis=1), df['Salary']
        self.ruleset: ClassificationRuleSet = load_classification_ruleset()
        self.ruleset.update(self.X, self.y, measure=measures.c2)

    def test_prediction_indicators(self):
        y_pred: np.ndarray = self.ruleset.predict(self.X)
        indicators: dict = calculate_for_classification(
            y_true=self.y, y_pred=y_pred
        )
        self.assertTrue(isinstance(indicators, dict))

    def test_prediction_indicators_only_on_covered_examples(self):
        self.ruleset.rules = self.ruleset.rules[int(
            len(self.ruleset.rules) / 10 * 8):]
        self.ruleset.set_default_conclusion_enabled(False)
        y_pred: np.ndarray = self.ruleset.predict(self.X)
        indicators_on_full_dataset = calculate_for_classification(
            y_true=self.y, y_pred=y_pred,
            calculate_only_for_covered_examples=False
        )
        indicators_on_covered_example = calculate_for_classification(
            y_true=self.y, y_pred=y_pred,
            calculate_only_for_covered_examples=True
        )
        self.assertGreater(
            indicators_on_covered_example['general']['Balanced_accuracy'],
            indicators_on_full_dataset['general']['Balanced_accuracy'],
            'Balanced accuracy should be better when calculated only on covered examples'
        )
