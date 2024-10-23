# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import json
import os
import unittest

import numpy as np
import pandas as pd
from decision_rules.core.ruleset import InvalidStateError
from decision_rules.serialization.utils import JSONSerializer
from decision_rules.survival.ruleset import SurvivalConclusion
from decision_rules.survival.ruleset import SurvivalRuleSet
from tests.helpers import compare_survival_prediction
from tests.loaders import load_resources_path


class TestSurvivalRuleSet(unittest.TestCase):

    def setUp(self) -> None:
        self.df = pd.read_csv(os.path.join(
            load_resources_path(), 'survival', 'BHS.csv'
        ))
        self.X = self.df.drop('survival_status', axis=1)
        self.y = self.df['survival_status'].astype(int).astype(str)

        ruleset_file_path: str = os.path.join(
            load_resources_path(), 'survival', 'BHS_ruleset.json'
        )
        with open(ruleset_file_path, 'r', encoding='utf-8') as file:
            self.ruleset: SurvivalRuleSet = JSONSerializer.deserialize(
                json.load(file),
                SurvivalRuleSet
            )

    def test_calculating_ibs(self):
        self.ruleset.update(self.X, self.y)
        ibs_calculated = self.ruleset.integrated_bier_score(self.X, self.y)
        ibs_rulekit = 0.08616902098503432

        self.assertEqual(
            ibs_calculated, ibs_rulekit,
            'Should calculate integrated_bier_score correctly'
        )

    def test_prediction(self):
        df = pd.read_csv(os.path.join(
            load_resources_path(), 'survival', 'bone-marrow.csv'
        ))
        X = df.drop('survival_status', axis=1)
        y = df['survival_status'].astype(int).astype(str)

        ruleset_file_path: str = os.path.join(
            load_resources_path(), 'survival', 'bone-marrow-survival-ruleset.json')
        with open(ruleset_file_path, 'r', encoding='utf-8') as file:
            ruleset: SurvivalRuleSet = JSONSerializer.deserialize(
                json.load(file),
                SurvivalRuleSet
            )
        coverage_matrix: np.ndarray = ruleset.update(X, y)

        prediction = ruleset.predict(X)

        with open(os.path.join(
            load_resources_path(), 'survival', 'rulekit-bone-marrow-prediction.json'
        ), 'r', encoding='utf-8') as file:
            rulekit_prediction = np.array(json.load(file))

        self.assertTrue(compare_survival_prediction(
            prediction, rulekit_prediction
        ),  'Prediction should be the same as rulekit prediction')

        prediction = ruleset.predict_using_coverage_matrix(
            coverage_matrix)

        self.assertTrue(compare_survival_prediction(
            prediction, rulekit_prediction
        ),  'Prediction should be the same as rulekit prediction')

    def test_different_prediction_strategies(self):
        coverage_matrix: np.ndarray = self.ruleset.update(self.X, self.y)

        for strategy in self.ruleset.prediction_strategies_choice.keys():
            self.ruleset.set_prediction_strategy(strategy)
            prediction = self.ruleset.predict(self.X)
            predition_cov_matrix = self.ruleset.predict_using_coverage_matrix(
                coverage_matrix
            )
            self.assertTrue(compare_survival_prediction(
                prediction, predition_cov_matrix
            ))

    def test_condition_importances(self):
        self.ruleset.update(self.X, self.y)

        for strategy in self.ruleset.prediction_strategies_choice.keys():
            self.ruleset.set_prediction_strategy(strategy)

            condition_importances = self.ruleset.calculate_condition_importances(
                self.X, self.y)
            attribute_importances = self.ruleset.calculate_attribute_importances(
                condition_importances)

            condition_importances_file_path: str = os.path.join(
                load_resources_path(), 'survival', 'BHS_condition_importances.json')
            with open(condition_importances_file_path, 'r', encoding='utf-8') as file:
                condition_importances_readed = json.load(file)

            attribute_importances_file_path: str = os.path.join(
                load_resources_path(), 'survival', 'BHS_attribute_importances.json')
            with open(attribute_importances_file_path, 'r', encoding='utf-8') as file:
                attribute_importances_readed = json.load(file)

            self.assertEqual(
                condition_importances, condition_importances_readed,
                'Condition importances should be the same as saveed before'
            )

            self.assertEqual(
                attribute_importances, attribute_importances_readed,
                'Attribute importances should be the same as saved before'
            )

    def test_local_exlainability(self):
        self.ruleset.update(self.X, self.y)

        for strategy in self.ruleset.prediction_strategies_choice.keys():
            self.ruleset.set_prediction_strategy(strategy)
            res = self.ruleset.local_explainability(self.X.iloc[0])
            rules_covering = res[0]
            kaplan_meier = res[1]

            self.assertEqual(len(rules_covering), 2,
                             'There should be 2 rules covering instance')
            self.assertTrue(all(kaplan_meier['probabilities']) >= 0 and all(
                kaplan_meier['probabilities']) <= 1)

    def test_survival_status_validation(self):
        self.y = self.y.astype(int)
        with self.assertRaises(ValueError, msg='Should fail for survival status of type different than string'):
            self.ruleset.update(self.X, self.y)

        self.y = self.y.astype(str)
        self.y.iloc[0] = '-1'
        with self.assertRaises(ValueError, msg='Should fail for survival status with values different than 0 and 1'):
            self.ruleset.update(self.X, self.y)

    def test_if_prediction_without_update_fails_with_meaningful_error(self):
        with self.assertRaises(
            InvalidStateError,
            msg='Should fail for prediction without update with meaningful error'
        ):
            self.ruleset.predict(self.X)

    def test_prediction_with_empty_default_conclusion(self):
        # remove one rule to leave some example uncovered
        self.ruleset.rules = self.ruleset.rules[1:2]
        self.ruleset.update(self.X, self.y)
        self.ruleset.set_default_conclusion_enabled(False)

        prediction: np.ndarray = self.ruleset.predict(self.X)
        self.assertTrue(
            any(e is None for e in prediction),
            'Prediction for some examples should be empty'
        )


if __name__ == '__main__':
    unittest.main()
