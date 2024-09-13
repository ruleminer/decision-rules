"""
Contains classification ruleset class.
"""
from __future__ import annotations

from typing import Callable
from typing import Type

import numpy as np
import pandas as pd
from decision_rules.classification.metrics import ClassificationRulesMetrics
from decision_rules.classification.prediction import VotingPredictionStrategy
from decision_rules.classification.rule import ClassificationConclusion
from decision_rules.classification.rule import ClassificationRule
from decision_rules.core.coverage import ClassificationCoverageInfodict
from decision_rules.core.coverage import Coverage
from decision_rules.core.metrics import AbstractRulesMetrics
from decision_rules.core.prediction import BestRulePredictionStrategy
from decision_rules.core.prediction import PredictionStrategy
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.importances._classification.attributes import \
    ClassificationRuleSetAttributeImportances
from decision_rules.importances._classification.conditions import \
    ClassificationRuleSetConditionImportances


class ClassificationRuleSet(AbstractRuleSet):
    """Classification ruleset allowing to perform prediction on data
    """

    def __init__(
        self,
        rules: list[ClassificationRule],
    ) -> None:
        """
        Args:
            rules (list[ClassificationRule]):
        """
        self.rules: list[ClassificationRule]
        super().__init__(rules)

    def get_metrics_object_instance(self) -> AbstractRulesMetrics:
        return ClassificationRulesMetrics(self.rules)

    def update_using_coverages(
        self,
        coverages_info: dict[str, ClassificationCoverageInfodict],
        measure: Callable[[Coverage], float],
        columns_names: list[str] = None,
    ):
        super().update_using_coverages(coverages_info, measure, columns_names)

    def update(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        measure: Callable[[Coverage], float]
    ) -> np.ndarray:
        coverage_matrix: np.ndarray = super().update(X_train, y_train, measure)
        return coverage_matrix

    def _update_majority_class(self):
        majority_class = pd.Series(
            self.train_P).sort_index().sort_values().index[-1]
        decision_attribute: str = (
            self.rules[0].conclusion.column_name
            if len(self.rules) > 0 else None
        )
        self.default_conclusion = ClassificationConclusion(
            value=majority_class,
            column_name=decision_attribute
        )
        self._stored_default_conclusion = self.default_conclusion

    def _calculate_P_N(self, y_uniques: np.ndarray, y_values_count: np.ndarray):  # pylint: disable=invalid-name
        all_counts: int = np.sum(y_values_count)
        all_decision_values_present_in_rules = [
            rule.conclusion.value for rule in self.rules]
        self.train_P = {
            value: 0 for value in all_decision_values_present_in_rules
        }
        self.train_N = {
            value: 0 for value in all_decision_values_present_in_rules
        }
        for i, value in enumerate(y_uniques):
            self.train_P[value] = y_values_count[i]
            self.train_N[value] = all_counts - y_values_count[i]
        self._update_majority_class()

    def calculate_condition_importances(
            self,
            X: pd.DataFrame,  # pylint: disable=invalid-name
            y: pd.Series,  # pylint: disable=invalid-name
            measure: Callable[[Coverage], float]
    ) -> dict[str, dict[str, float]]:
        condtion_importances_generator = ClassificationRuleSetConditionImportances(
            self)
        self.condition_importances = condtion_importances_generator.calculate_importances(
            X.to_numpy(), y.to_numpy(), measure)
        return self.condition_importances

    def calculate_attribute_importances(self, condition_importances: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        attributes_importances_generator = ClassificationRuleSetAttributeImportances()
        self.attribute_importances = attributes_importances_generator.calculate_importances_base_on_conditions(
            condition_importances)
        return self.attribute_importances

    def calculate_p_values(self, *args) -> list:
        metrics: AbstractRulesMetrics = self.get_metrics_object_instance()
        p_values = []
        for rule in self.rules:
            p_values.append(metrics.calculate_p_value(coverage=rule.coverage))
        return p_values

    @property
    def prediction_strategies_choice(self) -> dict[str, Type[PredictionStrategy]]:
        return {
            'vote': VotingPredictionStrategy,
            'best_rule': BestRulePredictionStrategy,
        }

    def get_default_prediction_strategy_class(self) -> Type[PredictionStrategy]:
        return VotingPredictionStrategy
