"""
Contains ConditionImportance class for determining importances of condtions in RuleSet.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np
from decision_rules.core.coverage import Coverage
from decision_rules.core.rule import AbstractCondition
from decision_rules.core.rule import AbstractRule
from decision_rules.core.ruleset import AbstractRuleSet


@dataclass
class ConditionImportance:
    def __init__(self, condition, quality) -> None:
        self.condition = condition
        self.quality = quality


class AbstractRuleSetConditionImportances(ABC):
    """Abstract ConditionImportance allowing to determine importances of condtions in RuleSet
    """

    def __init__(self, ruleset: AbstractRuleSet):
        """Constructor method
        """
        self.ruleset = deepcopy(ruleset)

    def calculate_importances(self, X: np.array, y: np.array, measure: Callable[[Coverage], float]) -> dict[str, dict[str, float]]:
        """Calculate importances of conditions in RuleSet
        """
        conditions_with_rules = self._get_conditions_with_rules(
            self.ruleset.rules)
        conditions_importances = self._calculate_conditions_importances(
            conditions_with_rules, X, y, measure)

        conditions_importances = self._prepare_importances(
            conditions_importances
        )

        return conditions_importances

    def _get_conditions_with_rules(self, rules: list[AbstractRule]) -> dict[AbstractCondition, list[AbstractRule]]:
        conditions_with_rules = defaultdict(list)
        for rule in rules:
            rule_conditions = rule.premise.subconditions
            for condition in rule_conditions:
                conditions_with_rules[condition].append(rule)

        return conditions_with_rules

    def _calculate_conditions_importances(self, conditions_with_rules: dict[str, list[AbstractRule]],  X: np.ndarray, y: np.ndarray, measure: Callable[[Coverage], float]) -> list[ConditionImportance]:
        conditions_importances = []
        for condition in conditions_with_rules.keys():
            sum = 0
            for rule in conditions_with_rules[condition]:
                sum += self._calculate_index_simplified(
                    condition, rule, X, y, measure)
            conditions_importances.append(ConditionImportance(condition, sum))

        return conditions_importances

    @abstractmethod
    def _calculate_index_simplified(self, condition: AbstractCondition, rule: AbstractRule, X: np.ndarray, y: np.ndarray, measure: Callable[[Coverage], float]) -> float:
        pass

    def _calculate_measure(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, measure: Callable[[Coverage], float]):
        return measure(rule.calculate_coverage(X, y))

    def _prepare_importances(self, conditions_importances: list[ConditionImportance]) -> list[dict]:

        conditions_importances_list = []

        for condition_importance in conditions_importances:
            attribute_indices = condition_importance.condition.attributes
            attribute_names = [self.ruleset.column_names[index]
                               for index in attribute_indices]
            condition_string = condition_importance.condition.to_string(
                columns_names=self.ruleset.column_names)

            conditions_importances_list.append({
                "condition": condition_string,
                "attributes": attribute_names,
                "importance": condition_importance.quality
            })

        conditions_importances_list = sorted(
            conditions_importances_list, key=lambda x: x["importance"], reverse=True)

        return conditions_importances_list
