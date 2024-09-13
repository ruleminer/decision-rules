"""
Contains ConditionImportance class for determining importances of condtions in RuleSet.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from decision_rules.conditions import CompoundCondition
from decision_rules.core.coverage import Coverage
from decision_rules.core.rule import AbstractCondition
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.importances._core import \
    AbstractRuleSetConditionImportances
from decision_rules.regression.rule import RegressionRule


class RegressionRuleSetConditionImportances(AbstractRuleSetConditionImportances):
    """Regression ConditionImportance allowing to determine importances of condtions in RuleSet
    """

    def _calculate_index_simplified(self, condition: AbstractCondition, rule: RegressionRule, X: np.ndarray, y: np.ndarray, measure: Callable[[Coverage], float]) -> float:
        rule_conditions = []
        rule_conditions.extend(rule.premise.subconditions)
        number_of_conditions = len(rule_conditions)
        rule_conditions.remove(condition)

        premise_without_evaluated_condition = CompoundCondition(
            subconditions=rule_conditions, logic_operator=rule.premise.logic_operator)

        rule_without_evaluated_condition = RegressionRule(
            premise_without_evaluated_condition,
            conclusion=rule.conclusion,
            column_names=rule.column_names
        )

        factor = 1.0 / number_of_conditions
        if len(rule_conditions) == 0:
            return factor * (
                self._calculate_measure(rule, X, y, measure)
                - self._calculate_measure(rule_without_evaluated_condition, X, y, measure)
            )
        else:
            premise_with_only_evaluated_condition = CompoundCondition(
                subconditions=[condition], logic_operator=rule.premise.logic_operator)

            rule_with_only_evaluated_condition = RegressionRule(
                premise_with_only_evaluated_condition,
                conclusion=rule.conclusion,
                column_names=rule.column_names
            )
            return factor * (
                self._calculate_measure(rule, X, y, measure)
                - self._calculate_measure(rule_without_evaluated_condition, X, y, measure)
                + self._calculate_measure(rule_with_only_evaluated_condition, X, y, measure)
            )
