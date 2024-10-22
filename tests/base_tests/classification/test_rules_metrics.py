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


if __name__ == '__main__':
    unittest.main()
