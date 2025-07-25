"""
Contains ConditionImportance class for determining importances of condtions in RuleSet.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from decision_rules.classification.rule import ClassificationRule
from decision_rules.conditions import CompoundCondition
from decision_rules.core.coverage import Coverage
from decision_rules.core.rule import AbstractCondition
from decision_rules.importances._core import AbstractRuleSetConditionImportances
from decision_rules.importances._core import ConditionImportance


class ClassificationRuleSetConditionImportances(AbstractRuleSetConditionImportances):
    """Classification ConditionImportance allowing to determine importances of condtions in RuleSet
    """

    def calculate_importances(self, X: np.array, y: np.array, measure: Callable[[Coverage], float]) -> dict[str, dict[str, float]]:
        """Calculate importances of conditions in RuleSet
        """
        rules_by_class = self._split_rules_by_decision_class(
            self.ruleset.rules)

        condition_importances_for_classes = {}

        for class_name, class_rules in rules_by_class.items():

            conditions_with_rules = self._get_conditions_with_rules(
                class_rules)
            conditions_importances = self._calculate_conditions_importances(
                conditions_with_rules, X, y, measure
            )
            condition_importances_for_classes[class_name] = conditions_importances

        conditions_importances = self._prepare_importances(
            condition_importances_for_classes
        )

        return conditions_importances

    def _split_rules_by_decision_class(self, rules: list[ClassificationRule]) -> dict[str, list[ClassificationRule]]:
        rules_by_class = {}
        for rule in rules:
            if rule.conclusion.value not in rules_by_class.keys():
                rules_by_class[rule.conclusion.value] = []
            rules_by_class[rule.conclusion.value].append(rule)
        return rules_by_class

    def _calculate_index_simplified(self, condition: AbstractCondition, rule: ClassificationRule, X: np.ndarray, y: np.ndarray, measure: Callable[[Coverage], float]) -> float:
        number_of_conditions = len(rule.premise.subconditions)
        premise_without_evaluated_condition = rule.premise.remove_condition_recursively(condition)

        factor = 1.0 / number_of_conditions
        if premise_without_evaluated_condition is None or len(premise_without_evaluated_condition.subconditions) == 0:
            empty_premise = CompoundCondition(subconditions=[], logic_operator=rule.premise.logic_operator)
            rule_without_evaluated_condition = ClassificationRule(
                empty_premise,
                conclusion=rule.conclusion,
                column_names=rule.column_names
            )
            return factor * (
                self._calculate_measure(rule, X, y, measure)
                - self._calculate_measure(rule_without_evaluated_condition, X, y, measure)
            )
        else:
            rule_without_evaluated_condition = ClassificationRule(
                premise_without_evaluated_condition,
                conclusion=rule.conclusion,
                column_names=rule.column_names
            )
            premise_with_only_evaluated_condition = CompoundCondition(
                subconditions=[condition], logic_operator=rule.premise.logic_operator)

            rule_with_only_evaluated_condition = ClassificationRule(
                premise_with_only_evaluated_condition,
                conclusion=rule.conclusion,
                column_names=rule.column_names
            )
            return factor * (
                self._calculate_measure(rule, X, y, measure)
                - self._calculate_measure(rule_without_evaluated_condition, X, y, measure)
                + self._calculate_measure(rule_with_only_evaluated_condition, X, y, measure)
            )

    def _prepare_importances(self, conditions_importances: dict[str, list[ConditionImportance]]) -> dict[str, list[dict]]:
        conditions_importances_sorted = {}

        for class_name, condition_importances_for_class_list in conditions_importances.items():
            conditions_importances_list = []

            for condition_importance in condition_importances_for_class_list:
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

            conditions_importances_sorted[class_name] = sorted(
                conditions_importances_list, key=lambda x: x["importance"], reverse=True)

        return conditions_importances_sorted
