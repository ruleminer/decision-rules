# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring,protected-access,invalid-name
import unittest

import pandas as pd

from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import NominalCondition
from decision_rules.survival.metrics import SurvivalRulesMetrics
from decision_rules.survival.rule import SurvivalConclusion
from decision_rules.survival.rule import SurvivalRule
from decision_rules.survival.ruleset import SurvivalRuleSet
from tests.base_tests.core.test_rules_metrics import BaseRulesMetricsTestCase


class TestSurvivalRulesMetrics(BaseRulesMetricsTestCase):

    def get_metrics_object_instance(self) -> SurvivalRulesMetrics:
        return SurvivalRulesMetrics(self.ruleset.rules)

    def setUp(self) -> None:
        self.X = pd.DataFrame(data=[
            ['a', 'a', 1],
            ['a', 'b', 2],
            ['b', 'b', 2],
            ['b', 'a', 2],
            ['b', 'b', 2],
        ], columns=['A', 'B', 'survival_time'])
        self.y = pd.Series(['1', '1', '1', '0', '0'], name='label')
        self.ruleset = SurvivalRuleSet(
            rules=[
                SurvivalRule(
                    premise=CompoundCondition(
                        subconditions=[
                            NominalCondition(
                                column_index=0,
                                value='a'
                            )
                        ]),
                    conclusion=SurvivalConclusion(
                        value=1, column_name='label'
                    ),
                    column_names=self.X.columns
                ),
                SurvivalRule(
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
                    conclusion=SurvivalConclusion(
                        value=0, column_name='label'),
                    column_names=self.X.columns
                ),
                SurvivalRule(
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
                    conclusion=SurvivalConclusion(
                        value=1, column_name='label'),
                    column_names=self.X.columns
                )
            ],
            survival_time_attr='survival_time'
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
        expected_all_unique_rule1 = 2

        # Rule 2:
        # Premise: A == 'b' and B == 'b'
        # Covers rows 2 and 4 (labels 0, 0)
        # Rule 3 also covers these rows
        expected_all_unique_rule2 = 0

        # Rule 3:
        # Premise: A == 'b' and B == 'b'
        # Covers rows 2 and 4 (labels 0, 0)
        # Rule 2 also covers these rows
        expected_all_unique_rule3 = 0

        # Calculation using the method
        unique_all_rule1 = metrics_object._calculate_uniquely_covered_examples(
            self.ruleset.rules[0], X_np, y_np, 'all'
        )
        unique_all_rule2 = metrics_object._calculate_uniquely_covered_examples(
            self.ruleset.rules[1], X_np, y_np, 'all'
        )
        unique_all_rule3 = metrics_object._calculate_uniquely_covered_examples(
            self.ruleset.rules[2], X_np, y_np, 'all'
        )

        self.assertEqual(
            unique_all_rule1,
            expected_all_unique_rule1,
            'Uniquely covered negative examples for rule 1 do not match expected value'
        )
        self.assertEqual(
            unique_all_rule2,
            expected_all_unique_rule2,
            'Uniquely covered negative examples for rule 1 do not match expected value'
        )
        self.assertEqual(
            unique_all_rule3,
            expected_all_unique_rule3,
            'Uniquely covered negative examples for rule 1 do not match expected value'
        )


if __name__ == '__main__':
    unittest.main()
