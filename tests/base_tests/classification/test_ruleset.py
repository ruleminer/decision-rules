# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import json
import os
import unittest

import numpy as np
import pandas as pd
from decision_rules import measures
from decision_rules.classification.rule import ClassificationConclusion
from decision_rules.classification.rule import ClassificationRule
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import NominalCondition
from decision_rules.core.exceptions import InvalidStateError
from decision_rules.serialization.utils import JSONSerializer
from tests.loaders import load_resources_path


class TestClassificationRuleSet(unittest.TestCase):

    def _prepare_prediction_dataset_with_numerical_labels(self) -> tuple[pd.DataFrame, pd.Series]:
        columns = ['a', 'b']
        y: np.ndarray = pd.Series([1, 0, 0, 0])
        X: pd.DataFrame = pd.DataFrame(
            [[0, 0], [1, 0], [1, 1], [1, 0]], columns=columns
        )
        return X, y

    def _prepare_ruleset_for_predicting_numerical_labels(
        self,
        column_names: list[str]
    ) -> ClassificationRuleSet:
        rules = []
        rule1 = ClassificationRule(
            CompoundCondition(subconditions=[
                NominalCondition(column_index=0, value=0),
                NominalCondition(column_index=1, value=1)
            ]),
            conclusion=ClassificationConclusion(value=1, column_name='class'),
            column_names=column_names
        )
        rules.append(rule1)
        rule2 = ClassificationRule(
            premise=NominalCondition(column_index=0, value=0),
            conclusion=ClassificationConclusion(value=0, column_name='class'),
            column_names=column_names
        )
        rules.append(rule2)
        return ClassificationRuleSet(rules)

    def _prepare_prediction_dataset_with_nominal_labels(self) -> tuple[pd.DataFrame, pd.Series]:
        columns = ['a', 'b']
        y: pd.Series = pd.Series(['a', 'b', 'b', 'b'])
        X: pd.DataFrame = pd.DataFrame(
            [[0, 0], [1, 0], [1, 1], [1, 0]], columns=columns
        )
        return X, y

    def _prepare_ruleset_for_predicting_nominal_labels(
        self,
        column_names: list[str]
    ) -> ClassificationRuleSet:
        rules = []

        rule1 = ClassificationRule(
            CompoundCondition(subconditions=[
                NominalCondition(column_index=0, value=0),
                NominalCondition(column_index=1, value=1)
            ]),
            conclusion=ClassificationConclusion(
                value='a', column_name='class'),
            column_names=column_names
        )
        rules.append(rule1)
        rule2 = ClassificationRule(
            premise=NominalCondition(column_index=0, value=0),
            conclusion=ClassificationConclusion(
                value='b', column_name='class'),
            column_names=column_names
        )
        rules.append(rule2)
        return ClassificationRuleSet(rules)

    def test_predict_on_numercial_labels(self):
        X, y = self._prepare_prediction_dataset_with_numerical_labels()
        ruleset = self._prepare_ruleset_for_predicting_numerical_labels(
            column_names=X.columns.tolist()
        )
        ruleset.update(X, y, measure=measures.accuracy)

        for strategy in ruleset.prediction_strategies_choice.keys():
            ruleset.set_prediction_strategy(strategy)
            prediction = ruleset.predict(X)
            self.assertTrue(
                np.array_equal(prediction, y),
                'Prediction should be the same as y in this example'
            )

    def test_predict_using_coverage_matrix_on_numercial_labels(self):
        X, y = self._prepare_prediction_dataset_with_numerical_labels()
        ruleset = self._prepare_ruleset_for_predicting_numerical_labels(
            column_names=X.columns.tolist()
        )
        coverage_matrix: np.ndarray = ruleset.update(
            X, y, measure=measures.accuracy)

        for strategy in ruleset.prediction_strategies_choice.keys():
            ruleset.set_prediction_strategy(strategy)
            prediction = ruleset.predict_using_coverage_matrix(coverage_matrix)
            self.assertTrue(
                np.array_equal(prediction, y),
                'Prediction should be the same as y in this example'
            )

    def test_predict_on_nominal_labels(self):
        X, y = self._prepare_prediction_dataset_with_nominal_labels()
        ruleset = self._prepare_ruleset_for_predicting_nominal_labels(
            column_names=X.columns.tolist()
        )
        ruleset.update(X, y, measure=measures.accuracy)

        for strategy in ruleset.prediction_strategies_choice.keys():
            ruleset.set_prediction_strategy(strategy)
            prediction = ruleset.predict(X)
            self.assertTrue(
                np.array_equal(prediction, y),
                'Prediction should be the same as y in this example'
            )

    def test_predict_using_coverage_matrix_on_nominal_labels(self):
        X, y = self._prepare_prediction_dataset_with_nominal_labels()
        ruleset = self._prepare_ruleset_for_predicting_nominal_labels(
            column_names=X.columns.tolist()
        )
        coverage_matrix: np.ndarray = ruleset.update(
            X, y, measure=measures.accuracy)

        for strategy in ruleset.prediction_strategies_choice.keys():
            ruleset.set_prediction_strategy(strategy)
            prediction = ruleset.predict_using_coverage_matrix(coverage_matrix)
            self.assertTrue(
                np.array_equal(prediction, y),
                'Prediction should be the same as y in this example'
            )

    def test_on_different_columns_order(self):
        X, y = self._prepare_prediction_dataset_with_nominal_labels()
        ruleset = self._prepare_ruleset_for_predicting_nominal_labels(
            column_names=X.columns.tolist()
        )
        ruleset.update(X, y, measure=measures.c2)

        first_prediction = ruleset.predict(X)
        X = X[reversed(X.columns.tolist())]
        second_prediction = ruleset.predict(X)

        self.assertTrue(
            np.array_equal(first_prediction, second_prediction),
            'RuleSet should predict the same no matter the column order.'
        )

    def test_prediction_with_empty_default_conclusion(self):
        X, y = self._prepare_prediction_dataset_with_nominal_labels()
        ruleset = self._prepare_ruleset_for_predicting_nominal_labels(
            column_names=X.columns.tolist()
        )
        # remove one rule to leave some example uncovered
        ruleset.rules = ruleset.rules[:-1]
        ruleset.update(X, y, measure=measures.c2)
        # set default conclusion to empty conclusion
        ruleset.set_default_conclusion_enabled(False)

        prediction: np.ndarray = ruleset.predict(X)
        self.assertTrue(
            any(e == '' for e in prediction),
            'Prediction for some examples should be empty'
        )

        ruleset.set_default_conclusion_enabled(True)

        prediction: np.ndarray = ruleset.predict(X)
        self.assertTrue(
            not any(e == '' for e in prediction),
            'Prediction for all examples should not be empty'
        )

    def test_set_enable_default_conclusion(self):
        X, y = self._prepare_prediction_dataset_with_nominal_labels()
        ruleset = self._prepare_ruleset_for_predicting_nominal_labels(
            column_names=X.columns.tolist()
        )
        ruleset: ClassificationRuleSet = self._prepare_ruleset_for_predicting_nominal_labels(
            column_names=X.columns.tolist()
        )
        self.assertTrue(
            ruleset.is_using_default_conclusion,
            'Default conclusion should be used by default'
        )
        with self.assertRaises(InvalidStateError):
            ruleset.set_default_conclusion_enabled(False)
        self.assertTrue(ruleset.is_using_default_conclusion)

        ruleset.update(X, y, measure=measures.c2)
        original_default_conclusion = ruleset.default_conclusion
        self.assertFalse(
            original_default_conclusion.is_empty(),
            'Default conclusion should not be empty'
        )

        ruleset.default_conclusion = ClassificationConclusion.make_empty(
            ruleset.decision_attribute
        )
        ruleset.set_default_conclusion_enabled(False)

        self.assertFalse(
            ruleset.is_using_default_conclusion,
            'Default conclusion should not be used'
        )

        ruleset.set_default_conclusion_enabled(True)
        self.assertEqual(
            original_default_conclusion,
            ruleset.default_conclusion,
            'Default conclusion should be enabled'
        )
        self.assertTrue(ruleset.is_using_default_conclusion,)

    def test_local_explainability(self):
        columns = ['a', 'b']
        y: np.ndarray = pd.Series(['a', 'b', 'b', 'b'])
        X: pd.DataFrame = pd.DataFrame(
            [[0, 0], [1, 0], [1, 1], [1, 0]], columns=columns
        )
        rules = []
        rules_that_cover_example_uuid = []

        # rule that covers example for local explainability
        # and have same conclusion as example class
        rule1 = ClassificationRule(
            CompoundCondition(subconditions=[
                NominalCondition(column_index=0, value=0),
                NominalCondition(column_index=1, value=0)
            ]),
            conclusion=ClassificationConclusion(
                value="a", column_name='class'),
            column_names=columns
        )
        rules.append(rule1)
        rules_that_cover_example_uuid.append(rule1.uuid)

        # rule that covers example for local explainability
        # and have different conclusion than example class
        rule2 = ClassificationRule(
            premise=NominalCondition(column_index=0, value=0),
            conclusion=ClassificationConclusion(
                value="b", column_name='class'),
            column_names=columns
        )
        rules.append(rule2)
        rules_that_cover_example_uuid.append(rule2.uuid)

        # rule that covers example for local explainability
        # and have same conclusion as example class
        rule3 = ClassificationRule(
            premise=NominalCondition(column_index=1, value=0),
            conclusion=ClassificationConclusion(
                value="a", column_name='class'),
            column_names=columns
        )
        rules.append(rule3)
        rules_that_cover_example_uuid.append(rule3.uuid)

        # rule that dosen't covers example for local explainability
        rule4 = ClassificationRule(
            premise=NominalCondition(column_index=1, value=1),
            conclusion=ClassificationConclusion(
                value="b", column_name='class'),
            column_names=columns
        )
        rules.append(rule4)

        ruleset = ClassificationRuleSet(rules)
        ruleset.update(X, y, measure=measures.precision)

        for strategy in ruleset.prediction_strategies_choice.keys():
            ruleset.set_prediction_strategy(strategy)
            y_pred = ruleset.predict(X)
            local_rules, prediction = ruleset.local_explainability(X.iloc[0])

            self.assertEqual(
                y_pred[0], prediction,
                'RuleSet prediction and prediction from local explainability should be the same'
            )
            self.assertEqual(
                sorted(rules_that_cover_example_uuid), sorted(local_rules),
                'Local explainability should return all rules that covers given example'
            )

    def test_condition_importances(self):
        importances_path = os.path.join(load_resources_path(), "importances")
        df = pd.read_csv(os.path.join(importances_path, "car.csv"), sep=";")
        y = df['class']
        X = df.drop('class', axis=1)

        ruleXAI_condition_importances = pd.read_csv(
            os.path.join(
                importances_path,
                "rulexai_condition_importances_for_car_measure_C2.csv"
            ),
            sep=";"
        )
        ruleXAI_condition_importances["acc | importances"] = ruleXAI_condition_importances["acc | importances"].astype(
            str)
        with open(os.path.join(importances_path, "car_ruleset.json"), "r") as fp:
            ruleset_json = json.load(fp)

        ruleset: ClassificationRuleSet = JSONSerializer.deserialize(
            ruleset_json, target_class=ClassificationRuleSet)
        ruleset.update(X, y, measure=measures.c2)

        for strategy in ruleset.prediction_strategies_choice.keys():
            ruleset.set_prediction_strategy(strategy)

            condition_importances = ruleset.calculate_condition_importances(
                X, y, measure=measures.c2)
            condition_importances_df = self._condition_importances_to_DataFrame(
                condition_importances)

            self.assertTrue(
                condition_importances_df.equals(ruleXAI_condition_importances),
                'Condition importances should be the same as in RuleXAI'
            )

    def _condition_importances_to_DataFrame(self, condition_importances):
        importances_df = pd.DataFrame()
        for class_name in condition_importances.keys():
            importances_df_tmp = pd.DataFrame()
            importances_df_tmp[class_name + " | conditions_names"] = pd.Series(
                [e['condition'] for e in condition_importances[class_name]]
            )
            importances_df_tmp[class_name + " | importances"] = pd.Series(
                [e['importance'] for e in condition_importances[class_name]]
            )
            importances_df = pd.concat(
                [importances_df, importances_df_tmp], ignore_index=False, axis=1
            )
        importances_df = importances_df.round(5)
        return importances_df.replace(np.nan, "-").astype(str)


if __name__ == '__main__':
    unittest.main()
