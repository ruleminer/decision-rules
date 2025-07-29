"""
Contains ConditionImportance class for determining importances of condtions in RuleSet.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from decision_rules.conditions import CompoundCondition
from decision_rules.core.coverage import Coverage
from decision_rules.core.rule import AbstractCondition
from decision_rules.core.rule import AbstractRule
from decision_rules.importances._core import AbstractRuleSetConditionImportances
from decision_rules.importances._core import ConditionImportance
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator
from decision_rules.survival.rule import SurvivalRule


class SurvivalRuleSetConditionImportances(AbstractRuleSetConditionImportances):
    """Survival ConditionImportance allowing to determine importances of condtions in RuleSet
    """

    def calculate_importances(self, X: np.array, y: np.array) -> dict[str, dict[str, float]]:
        """Calculate importances of conditions in RuleSet
        """
        conditions_with_rules = self._get_conditions_with_rules(
            self.ruleset.rules)
        conditions_importances = self._calculate_conditions_importances(
            conditions_with_rules, X, y)

        conditions_importances = self._prepare_importances(
            conditions_importances
        )

        return conditions_importances

    def _calculate_conditions_importances(self, conditions_with_rules: dict[str, list[AbstractRule]],  X: np.ndarray, y: np.ndarray) -> list[ConditionImportance]:
        conditions_importances = []
        for condition in conditions_with_rules.keys():
            sum = 0
            for rule in conditions_with_rules[condition]:
                sum += self._calculate_index_simplified(
                    condition, rule, X, y)
            conditions_importances.append(ConditionImportance(condition, sum))

        return conditions_importances

    def _calculate_index_simplified(self, condition: AbstractCondition, rule: SurvivalRule, X: np.ndarray, y: np.ndarray) -> float:
        number_of_conditions = len(rule.premise.subconditions)
        premise_without_evaluated_condition = rule.premise.remove_condition_recursively(condition)

        factor = 1.0 / number_of_conditions
        if premise_without_evaluated_condition is None or len(premise_without_evaluated_condition.subconditions) == 0:
            empty_premise = CompoundCondition(subconditions=[], logic_operator=rule.premise.logic_operator)
            rule_without_evaluated_condition = SurvivalRule(
                empty_premise,
                conclusion=rule.conclusion,
                column_names=rule.column_names,
                survival_time_attr=rule.survival_time_attr
            )
            return factor * (
                self._calculate_measure(rule, X, y)
                - self._calculate_measure(rule_without_evaluated_condition, X, y)
            )
        else:
            rule_without_evaluated_condition = SurvivalRule(
                premise_without_evaluated_condition,
                conclusion=rule.conclusion,
                column_names=rule.column_names,
                survival_time_attr=rule.survival_time_attr
            )

            premise_with_only_evaluated_condition = CompoundCondition(
                subconditions=[condition], logic_operator=rule.premise.logic_operator)

            rule_with_only_evaluated_condition = SurvivalRule(
                premise_with_only_evaluated_condition,
                conclusion=rule.conclusion,
                column_names=rule.column_names,
                survival_time_attr=rule.survival_time_attr
            )
            return factor * (
                self._calculate_measure(rule, X, y)
                - self._calculate_measure(rule_without_evaluated_condition, X, y)
                + self._calculate_measure(rule_with_only_evaluated_condition, X, y)
            )

    def _calculate_measure(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray):
        covered_examples_indexes = np.where(
            rule.premise._calculate_covered_mask(X))[0]
        uncovered_examples_indexes = np.where(
            rule.premise._calculate_uncovered_mask(X))[0]
        log_rank = KaplanMeierEstimator.log_rank(X[:, rule.survival_time_attr_idx],
                                                 y, covered_examples_indexes, uncovered_examples_indexes)
        return log_rank
