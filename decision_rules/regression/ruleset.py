"""
Contains regression ruleset class.
"""
from __future__ import annotations

from typing import Callable
from typing import Type

import numpy as np
import pandas as pd
from decision_rules.core.coverage import Coverage
from decision_rules.core.coverage import RegressionCoverageInfodict
from decision_rules.core.metrics import AbstractRulesMetrics
from decision_rules.core.prediction import BestRulePredictionStrategy
from decision_rules.core.prediction import PredictionStrategy
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.importances._regression.attributes import \
    RegressionRuleSetAttributeImportances
from decision_rules.importances._regression.conditions import \
    RegressionRuleSetConditionImportances
from decision_rules.regression.metrics import RegressionRulesMetrics
from decision_rules.regression.prediction import VotingPredictionStrategy
from decision_rules.regression.rule import RegressionConclusion
from decision_rules.regression.rule import RegressionRule


class RegressionRuleSet(AbstractRuleSet):
    """Regression ruleset allowing to perform prediction on data
    """

    def __init__(
        self,
        rules: list[RegressionRule],
    ) -> None:
        """
        Args:
            rules (list[RegressionRule]):
        """
        self.rules: list[RegressionRule]
        super().__init__(rules)
        self._y_train_median: float = None
        self.decision_attribute: str = (
            self.rules[0].conclusion.column_name
            if len(rules) > 0 else None
        )
        self.default_conclusion = RegressionConclusion(
            value=0.0,
            low=0.0,
            high=0.0,
            column_name=self.decision_attribute
        )
        self._stored_default_conclusion = self.default_conclusion

    def get_metrics_object_instance(self) -> AbstractRulesMetrics:
        return RegressionRulesMetrics(self.rules)

    @property
    def y_train_median(self) -> float:
        """
        Returns:
            float: label's median value on train dataset
        """
        return self._y_train_median

    def update_using_coverages(
        self,
        coverages_info: dict[str, RegressionCoverageInfodict],
        measure: Callable[[Coverage], float],
        columns_names: list[str] = None
    ):
        self.default_conclusion = RegressionConclusion(
            value=self._y_train_median,
            low=self._y_train_median,
            high=self._y_train_median,
            column_name=self.rules[0].conclusion.column_name
        )
        self._stored_default_conclusion = self.default_conclusion
        super().update_using_coverages(coverages_info, measure, columns_names)

        # update some additional regression specific informations
        for rule in self.rules:
            coverage_info: RegressionCoverageInfodict = coverages_info[rule.uuid]
            rule.conclusion.train_covered_y_mean = coverage_info['train_covered_y_mean']
            rule.conclusion.train_covered_y_std = coverage_info['train_covered_y_std']

    def update(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        measure: Callable[[Coverage], float]
    ) -> np.ndarray:
        self._y_train_median = y_train.median()
        self.default_conclusion = RegressionConclusion(
            value=self._y_train_median,
            low=self._y_train_median,
            high=self._y_train_median,
            column_name=self.decision_attribute
        )
        self._stored_default_conclusion = self.default_conclusion
        return super().update(X_train, y_train, measure)

    def _calculate_P_N(self, y_uniques: np.ndarray, y_values_count: np.ndarray):  # pylint: disable=invalid-name
        return

    def calculate_condition_importances(
            self,
            X: pd.DataFrame,  # pylint: disable=invalid-name
            y: pd.Series,  # pylint: disable=invalid-name
            measure: Callable[[Coverage], float]
    ) -> dict[str, dict[str, float]]:
        condition_importances_generator = RegressionRuleSetConditionImportances(
            self)
        self.condition_importances = condition_importances_generator.calculate_importances(
            X.to_numpy(), y.to_numpy(), measure)
        return self.condition_importances

    def calculate_attribute_importances(self, condition_importances: dict[str, float]) -> dict[str, float]:
        attributes_importances_generator = RegressionRuleSetAttributeImportances()
        self.attribute_importances = attributes_importances_generator.calculate_importances_base_on_conditions(
            condition_importances)
        return self.attribute_importances

    def calculate_p_values(self, y: np.ndarray) -> list:
        metrics: AbstractRulesMetrics = self.get_metrics_object_instance()
        p_values = []
        for rule in self.rules:
            p_values.append(metrics.calculate_p_value(rule=rule, y=y))
        return p_values

    @property
    def prediction_strategies_choice(self) -> dict[str, Type[PredictionStrategy]]:
        return {
            'vote': VotingPredictionStrategy,
            'best_rule': BestRulePredictionStrategy,
        }

    def get_default_prediction_strategy_class(self) -> Type[PredictionStrategy]:
        return VotingPredictionStrategy
