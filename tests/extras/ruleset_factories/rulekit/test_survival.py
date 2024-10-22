# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import json
import os
import unittest
from typing import Tuple

import numpy as np
import pandas as pd
from decision_rules.ruleset_factories import ruleset_factory
from decision_rules.serialization import JSONSerializer
from decision_rules.survival.ruleset import SurvivalRuleSet
from rulekit.survival import SurvivalRules

from tests.loaders import load_dataset_to_x_y, load_ruleset_factories_resources_path
from .helpers import compare_survival_prediction


class TestSurvivalRuleSet(unittest.TestCase):

    rulekit_model: SurvivalRules
    dataset_path = "survival/bone-marrow.csv"

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> SurvivalRules:
        rulekit_model = SurvivalRules(survival_time_attr="survival_time")
        rulekit_model.fit(X, y)
        return rulekit_model

    def test_if_factory_produces_correct_and_working_ruleset(self):
        X, y = load_dataset_to_x_y(self.dataset_path, y_col="survival_status")
        y = y.astype(str)
        model = self.train_model(X, y)

        ruleset: SurvivalRuleSet = ruleset_factory(model, X, y)
        ruleset.predict(X)

        self.assertEqual(
            len(model.model.rules),
            len(ruleset.rules),
            "RuleSet should contain the same number of rules as original RuleKit model",
        )

    def test_if_prediction_same_as_rulekit(self):
        X, y = load_dataset_to_x_y(self.dataset_path, y_col="survival_status")
        y = y.astype(str)
        model = self.train_model(X, y)

        ruleset: SurvivalRuleSet = ruleset_factory(model, X, y)
        y_pred: np.ndarray = ruleset.predict(X)
        y_pred_rulekit: np.ndarray = model.predict(X)

        self.assertTrue(
            compare_survival_prediction(y_pred, y_pred_rulekit),
            "RuleSet should predict the same as original RuleKit model",
        )

    def test_serialization(self):
        X, y = load_dataset_to_x_y(self.dataset_path, y_col="survival_status")
        y = y.astype(str)
        model = self.train_model(X, y)

        ruleset: SurvivalRuleSet = ruleset_factory(model, X, y)
        ruleset_serialized = JSONSerializer.serialize(ruleset)
        with open(
            self._get_dataset_ruleset_path(),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(ruleset_serialized, indent=2))
        ruleset_deserialized: SurvivalRuleSet = JSONSerializer.deserialize(
            ruleset_serialized, SurvivalRuleSet
        )
        ruleset_deserialized.update(X, y)

        y_pred: np.ndarray = ruleset.predict(X)
        y_pred_deserialized: np.ndarray = ruleset_deserialized.predict(X)
        self.assertTrue(
            compare_survival_prediction(y_pred, y_pred_deserialized),
            "Deserialized RuleSet should predict the same as original one",
        )

    @classmethod
    def _get_dataset_ruleset_path(cls) -> str:
        rule_dir = load_ruleset_factories_resources_path()
        return os.path.join(rule_dir, "bone-marrow-survival-ruleset.json")

    @classmethod
    def tearDownClass(cls):
        ruleset_path = cls._get_dataset_ruleset_path()
        if os.path.exists(ruleset_path):
            os.remove(ruleset_path)
