# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring,protected-access,invalid-name
import unittest

import pandas as pd
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import NominalCondition
from decision_rules.survival.metrics import SurvivalRulesMetrics
from decision_rules.survival.rule import SurvivalConclusion
from decision_rules.survival.rule import SurvivalRule
from decision_rules.survival.ruleset import SurvivalRuleSet
from tests.core.test_rules_metrics import BaseRulesMetricsTestCase


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


if __name__ == '__main__':
    unittest.main()
