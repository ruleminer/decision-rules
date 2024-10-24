# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring,protected-access,invalid-name
import unittest

import pandas as pd
from decision_rules.classification.metrics import ClassificationRulesMetrics
from decision_rules.classification.rule import ClassificationConclusion
from decision_rules.classification.rule import ClassificationRule
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import NominalCondition
from tests.base_tests.core.test_rules_metrics import BaseRulesMetricsTestCase


class TestClassificationRulesMetrics(BaseRulesMetricsTestCase):

    def get_metrics_object_instance(self) -> ClassificationRulesMetrics:
        return ClassificationRulesMetrics(self.ruleset.rules)

    def setUp(self) -> None:
        self.X = pd.DataFrame(data=[
            ['a', 'a'],
            ['a', 'b'],
            ['b', 'b'],
            ['b', 'a'],
            ['b', 'b'],
        ], columns=['A', 'B'])
        self.y = pd.Series([1, 1, 0, 0, 0], name='label')
        self.ruleset = ClassificationRuleSet(
            rules=[
                ClassificationRule(
                    premise=CompoundCondition(
                        subconditions=[
                            NominalCondition(
                                column_index=0,
                                value='a'
                            )
                        ]),
                    conclusion=ClassificationConclusion(
                        value=1, column_name='label'
                    ),
                    column_names=self.X.columns
                ),
                ClassificationRule(
                    premise=CompoundCondition(
                        subconditions=[
                            NominalCondition(
                                column_index=0,
                                value='b'
                            ),
                            NominalCondition(
                                column_index=1,
                                value='b'
                            )
                        ]),
                    conclusion=ClassificationConclusion(
                        value=0, column_name='label'
                    ),
                    column_names=self.X.columns
                ),
                ClassificationRule(
                    premise=CompoundCondition(
                        subconditions=[
                            NominalCondition(
                                column_index=0,
                                value='b'
                            ),
                            NominalCondition(
                                column_index=1,
                                value='b'
                            )
                        ]),
                    conclusion=ClassificationConclusion(
                        value=1, column_name='label'
                    ),
                    column_names=self.X.columns
                )
            ]
        )
        self._original_calculate_covered_mask = NominalCondition._calculate_covered_mask

    def tearDown(self) -> None:
        NominalCondition._calculate_covered_mask = self._original_calculate_covered_mask

    def test_calculate_uniquely_covered_examples_with_manual_values(self):
        metrics_object = self.get_metrics_object_instance()
        X_np = self.X.values
        y_np = self.y.values

        # Rule 1:
        # Premise: A == 'a'
        # Covers rows 0 and 1 (labels 1, 1)
        # Other rules do not cover these rows
        expected_p_unique_rule1 = 2 
        expected_n_unique_rule1 = 0 
        expected_all_unique_rule1 = 2

        # Rule 2:
        # Premise: A == 'b' and B == 'b'
        # Covers rows 2 and 4 (labels 0, 0)
        # Rule 3 also covers these rows
        expected_p_unique_rule2 = 0 
        expected_n_unique_rule2 = 0 
        expected_all_unique_rule2 = 0

        # Rule 3:
        # Premise: A == 'b' and B == 'b'
        # Covers rows 2 and 4 (labels 0, 0)
        # Rule 2 also covers these rows 
        expected_p_unique_rule3 = 0  
        expected_n_unique_rule3 = 0 
        expected_all_unique_rule3 = 0

        # Calculation using the method
        unique_positive_rule1 = metrics_object._calculate_uniquely_covered_examples(
            self.ruleset.rules[0], X_np, y_np, 'positive'
        )
        unique_negative_rule1 = metrics_object._calculate_uniquely_covered_examples(
            self.ruleset.rules[0], X_np, y_np, 'negative'
        )
        unique_all_rule1 = metrics_object._calculate_uniquely_covered_examples(
            self.ruleset.rules[0], X_np, y_np, 'all'
        )
        unique_positive_rule2 = metrics_object._calculate_uniquely_covered_examples(
            self.ruleset.rules[1], X_np, y_np, 'positive'
        )
        unique_negative_rule2 = metrics_object._calculate_uniquely_covered_examples(
            self.ruleset.rules[1], X_np, y_np, 'negative'
        )
        unique_all_rule2 = metrics_object._calculate_uniquely_covered_examples(
            self.ruleset.rules[1], X_np, y_np, 'all'
        )
        unique_positive_rule3 = metrics_object._calculate_uniquely_covered_examples(
            self.ruleset.rules[2], X_np, y_np, 'positive'
        )
        unique_negative_rule3 = metrics_object._calculate_uniquely_covered_examples(
            self.ruleset.rules[2], X_np, y_np, 'negative'
        )
        unique_all_rule3 = metrics_object._calculate_uniquely_covered_examples(
            self.ruleset.rules[2], X_np, y_np, 'all'
        )

        self.assertEqual(
            unique_positive_rule1,
            expected_p_unique_rule1,
            'Uniquely covered positive examples for rule 1 do not match expected value'
        )
        self.assertEqual(
            unique_negative_rule1,
            expected_n_unique_rule1,
            'Uniquely covered negative examples for rule 1 do not match expected value'
        )
        self.assertEqual(
            unique_all_rule1,
            expected_all_unique_rule1,
            'Uniquely covered negative examples for rule 1 do not match expected value'
        )
        self.assertEqual(
            unique_positive_rule2,
            expected_p_unique_rule2,
            'Uniquely covered positive examples for rule 2 do not match expected value'
        )
        self.assertEqual(
            unique_negative_rule2,
            expected_n_unique_rule2,
            'Uniquely covered negative examples for rule 2 do not match expected value'
        )
        self.assertEqual(
            unique_all_rule2,
            expected_all_unique_rule2,
            'Uniquely covered negative examples for rule 1 do not match expected value'
        )
        self.assertEqual(
            unique_positive_rule3,
            expected_p_unique_rule3,
            'Uniquely covered positive examples for rule 3 do not match expected value'
        )
        self.assertEqual(
            unique_negative_rule3,
            expected_n_unique_rule3,
            'Uniquely covered negative examples for rule 3 do not match expected value'
        )
        self.assertEqual(
            unique_all_rule3,
            expected_all_unique_rule3,
            'Uniquely covered negative examples for rule 1 do not match expected value'
        )

    def test_calculate_pos_and_neg_unique_with_manual_values(self):
        metrics_object = self.get_metrics_object_instance()
        X_np = self.X.values
        y_np = self.y.values

        # Rule 1:
        # Premise: A == 'a'
        # Covers rows 0 and 1 (labels 1, 1)
        # Prediction: 1
        # Other rules do not cover these rows
        expected_pos_unique_rule1 = 2
        expected_neg_unique_rule1 = 0 

        # Rule 2:
        # Premise: A == 'b' and B == 'b'
        # Covers rows 2 and 4 (labels 0, 0)
        # Rule 3 also covers these same rows but negative
        expected_pos_unique_rule2 = 2  
        expected_neg_unique_rule2 = 0  

        # Rule 3:
        # Premise: A == 'b' and B == 'b'
        # Covers rows 2 and 4 (labels 0, 0)
        # Rule 2 covers these rows appropriately, so Rule 3 uniquely covers them inappropriately
        expected_pos_unique_rule3 = 0 
        expected_neg_unique_rule3 = 2 

        # Calculation using the method
        unique_in_positive_rule1 = metrics_object._calculate_uniquely_covered_examples_in_pos_and_neg(
            self.ruleset.rules[0], X_np, y_np, 'positive'
        )
        unique_in_negative_rule1 = metrics_object._calculate_uniquely_covered_examples_in_pos_and_neg(
            self.ruleset.rules[0], X_np, y_np, 'negative'
        )
        unique_in_positive_rule2 = metrics_object._calculate_uniquely_covered_examples_in_pos_and_neg(
            self.ruleset.rules[1], X_np, y_np, 'positive'
        )
        unique_in_negative_rule2 = metrics_object._calculate_uniquely_covered_examples_in_pos_and_neg(
            self.ruleset.rules[1], X_np, y_np, 'negative'
        )
        unique_in_positive_rule3 = metrics_object._calculate_uniquely_covered_examples_in_pos_and_neg(
            self.ruleset.rules[2], X_np, y_np, 'positive'
        )
        unique_in_negative_rule3 = metrics_object._calculate_uniquely_covered_examples_in_pos_and_neg(
            self.ruleset.rules[2], X_np, y_np, 'negative'
        )

        self.assertEqual(
            unique_in_positive_rule1,
            expected_pos_unique_rule1,
            'Uniquely covered positive examples for rule 1 do not match expected value'
        )
        self.assertEqual(
            unique_in_negative_rule1,
            expected_neg_unique_rule1,
            'Uniquely covered negative examples for rule 1 do not match expected value'
        )
        self.assertEqual(
            unique_in_positive_rule2,
            expected_pos_unique_rule2,
            'Uniquely covered positive examples for rule 2 do not match expected value'
        )
        self.assertEqual(
            unique_in_negative_rule2,
            expected_neg_unique_rule2,
            'Uniquely covered negative examples for rule 2 do not match expected value'
        )
        self.assertEqual(
            unique_in_positive_rule3,
            expected_pos_unique_rule3,
            'Uniquely covered positive examples for rule 3 do not match expected value'
        )
        self.assertEqual(
            unique_in_negative_rule3,
            expected_neg_unique_rule3,
            'Uniquely covered negative examples for rule 3 do not match expected value'
        )

if __name__ == '__main__':
    unittest.main()
