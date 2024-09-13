# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring,protected-access,invalid-name
import unittest

import numpy as np
import pandas as pd
from decision_rules.conditions import NominalCondition
from decision_rules.core.metrics import AbstractRulesMetrics
from decision_rules.core.ruleset import AbstractRuleSet
from tests.helpers import check_if_any_of_dict_value_is_nan


def skip_if_base_class(func):
    def inner(self, *args, **kwargs):
        if self.__class__ == BaseRulesMetricsTestCase:
            self.skipTest('')

        func(self, *args, **kwargs)
    return inner


class BaseRulesMetricsTestCase(unittest.TestCase):

    X: pd.DataFrame
    y: pd.Series
    ruleset: AbstractRuleSet

    def get_metrics_object_instance(self) -> AbstractRulesMetrics:

        raise NotImplementedError('Not implemented')

    @skip_if_base_class
    def test_if_calculate_rules_metrics_calculates_coveres_multiple_times(self):
        conditions_count = sum(
            [len(rule.premise.subconditions) for rule in self.ruleset.rules]
        )
        covered_mask_calucation_count: int = {'count': 0}
        _calculate_covered_mask = NominalCondition._calculate_covered_mask

        def mock_calculate_covered_mask(self, X: np.ndarray) -> np.ndarray:
            covered_mask_calucation_count['count'] += 1
            return _calculate_covered_mask(self, X)
        NominalCondition._calculate_covered_mask = mock_calculate_covered_mask

        self.ruleset.calculate_rules_metrics(self.X, self.y)
        self.assertEqual(
            covered_mask_calucation_count['count'], conditions_count,
            'Calculation of covered_mask should be called only once for each rule premise'
        )

    @skip_if_base_class
    def test_calculate_all_metrics(self):
        metrics_values: dict = self.ruleset.calculate_rules_metrics(
            self.X, self.y)
        self.assertListEqual(
            list(metrics_values.keys()),
            [rule.uuid for rule in self.ruleset.rules],
            'Metrics should be calculated for every rule in ruleset.'
        )
        for rule_metrics in metrics_values.values():
            self.assertListEqual(
                list(rule_metrics.keys()),
                self.get_metrics_object_instance().supported_metrics,
                'All possible metrics should be calculated if not metrics names are specified'
            )
            self.assertFalse(
                check_if_any_of_dict_value_is_nan(rule_metrics),
                'Some of calculated rule metrics are nan'
            )

    @skip_if_base_class
    def test_calculate_specified_metrics(self):
        specified_metrics: list[str] = self.get_metrics_object_instance(
        ).supported_metrics[0:3]
        metrics_values: dict = self.ruleset.calculate_rules_metrics(
            self.X, self.y, metrics_to_calculate=specified_metrics
        )
        self.assertListEqual(
            list(metrics_values.keys()),
            [rule.uuid for rule in self.ruleset.rules],
            'Metrics should be calculated for every rule in ruleset.'
        )
        for rule_metrics in metrics_values.values():
            self.assertListEqual(
                list(rule_metrics.keys()),
                specified_metrics,
                'Only specified metrics should be calculated'
            )
            self.assertFalse(
                check_if_any_of_dict_value_is_nan(rule_metrics),
                'Some of calculated rule metrics are nan'
            )

    @skip_if_base_class
    def test_calculate_unsupported_metrics(self):
        with self.assertRaises(ValueError):
            self.ruleset.calculate_rules_metrics(
                self.X, self.y, metrics_to_calculate=[
                    'non_supported_metric_name']
            )
