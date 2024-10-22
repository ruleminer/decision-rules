# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring,protected-access,invalid-name
import unittest

import pandas as pd
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import NominalCondition
from decision_rules.regression.metrics import RegressionRulesMetrics
from decision_rules.regression.rule import RegressionConclusion
from decision_rules.regression.rule import RegressionRule
from decision_rules.regression.ruleset import RegressionRuleSet
from tests.base_tests.core.test_rules_metrics import BaseRulesMetricsTestCase


class TestRegressionRulesMetrics(BaseRulesMetricsTestCase):

    def get_metrics_object_instance(self) -> RegressionRulesMetrics:
        return RegressionRulesMetrics(self.ruleset.rules)

    def setUp(self) -> None:
        self.X = pd.DataFrame(data=[
            ['a', 'a'],
            ['a', 'b'],
            ['b', 'b'],
            ['b', 'a'],
            ['b', 'b'],
        ], columns=['A', 'B'])
        self.y = pd.Series([1, 1, 0, 0, 0], name='label')
        self.ruleset = RegressionRuleSet(
            rules=[
                RegressionRule(
                    premise=CompoundCondition(
                        subconditions=[
                            NominalCondition(
                                column_index=0,
                                value='a'
                            )
                        ]),
                    conclusion=RegressionConclusion(
                        value=1, column_name='label'
                    ),
                    column_names=self.X.columns
                ),
                RegressionRule(
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
                    conclusion=RegressionConclusion(
                        value=0,
                        low=0,
                        high=1,
                        column_name='label'
                    ),
                    column_names=self.X.columns
                ),
                RegressionRule(
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
                    conclusion=RegressionConclusion(
                        value=1,
                        low=0,
                        high=2,
                        column_name='label'
                    ),
                    column_names=self.X.columns
                )
            ]
        )
        self._original_calculate_covered_mask = NominalCondition._calculate_covered_mask

    def tearDown(self) -> None:
        NominalCondition._calculate_covered_mask = self._original_calculate_covered_mask


if __name__ == '__main__':
    unittest.main()
