from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Iterable, Optional

import packaging.version
import pandas as pd

from decision_rules.conditions import (AbstractCondition, CompoundCondition,
                                       ElementaryCondition, LogicOperators,
                                       NominalCondition)
from decision_rules.core.coverage import Coverage
from decision_rules.core.rule import AbstractConclusion, AbstractRule
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.helpers import get_measure_function_by_name
from decision_rules.ruleset_factories._factories.abstract_factory import \
    AbstractFactory

MINIMUM_RULEKIT_VERSION: str = "2.1.21"

def check_if_rulekit_is_installed_and_correct_version():
    try:
        from rulekit import __VERSION__
    except ImportError as e:
        raise ImportError(
            "The 'ruleset_factories' extra requires additional 'rulekit' "
            "dependencies. Please install them with: `pip install rulekit`"
        ) from e
    actual_version = packaging.version.parse(__VERSION__)
    required_version = packaging.version.parse(MINIMUM_RULEKIT_VERSION)
    if actual_version < required_version:
        raise ImportError(
            "The 'ruleset_factories' extra requires rulekit version 2.1.21 or higher. "
            f"Currently installed version is { __VERSION__}. "
            "Please install it with: `pip install rulekit>=2.1.21`"
        )


class AbstractRuleKitRuleSetFactory(AbstractFactory):

    def __init__(self) -> None:

        self.column_indices: dict = None
        self.columns_names: list[str] = None
        self.labels_values: Iterable[Any] = None

    def make(self, model, X_train: pd.DataFrame, y_train: pd.Series) -> AbstractRuleSet:
        self.labels_values = y_train.unique()
        self.columns_names = X_train.columns.tolist()
        self.column_indices = {
            column_name: i for i, column_name in enumerate(self.columns_names)
        }
        ruleset = self._make_ruleset(
            [self._make_rule(rule) for rule in model.model.rules]
        )
        self._calculate_P_N(model, ruleset)
        ruleset.column_names = self.columns_names
        ruleset.decision_attribute = y_train.name

        ruleset.update(X_train, y_train, measure=self._get_voting_measure(model))
        return ruleset

    def _get_voting_measure(
        self,
        model,
    ) -> Callable[[Coverage], float]:
        tmp: Enum = model.get_params()["voting_measure"]
        voting_measure_name: str = tmp.value
        return get_measure_function_by_name(voting_measure_name)

    @abstractmethod
    def _calculate_P_N(self, model, ruleset: AbstractRuleSet):
        pass

    def _make_compound_condition(
        self, parent: Optional[CompoundCondition], java_object: Any
    ) -> CompoundCondition:
        if "OR" in str(java_object.toString()):
            operator = LogicOperators.ALTERNATIVE
        else:
            operator = LogicOperators.CONJUNCTION

        condition = CompoundCondition(subconditions=[], logic_operator=operator)
        subconditions = java_object.getSubconditions()
        for subcondition in subconditions:
            self._make_condition(condition, subcondition)
        condition.subconditions.reverse()

        if len(condition.subconditions) == 1:
            # if only one condition, then return its only subcondition
            condition = condition.subconditions[0]

        parent.subconditions.append(condition)
        return condition

    def _make_nominal_condition(
        self,
        parent: Optional[CompoundCondition],
        java_object: Any,
        negated: bool = False,
    ) -> NominalCondition:
        value_set = java_object.getValueSet()
        condition = NominalCondition(
            column_index=self.column_indices[java_object.getAttribute()],
            value=str(value_set.getMapping().get(int(value_set.getValue()))),
        )
        condition.negated = negated
        parent.subconditions.append(condition)
        return condition

    def _make_numerical_condition(
        self, parent: Optional[CompoundCondition], java_object: Any
    ) -> NominalCondition:
        value_set = java_object.getValueSet()
        value_set_str = str(value_set)
        left = float(value_set.getLeft())
        if left == -value_set.INF:
            left = float("-inf")
        right = float(value_set.getRight())
        if right == value_set.INF:
            right = float("inf")
        condition = ElementaryCondition(
            column_index=self.column_indices[java_object.getAttribute()],
            left=left,
            right=right,
            left_closed=value_set_str.startswith("<"),
            right_closed=value_set_str.endswith(">"),
        )
        parent.subconditions.append(condition)
        return condition

    def _make_elementary_condition(
        self, parent: Optional[CompoundCondition], java_object: Any
    ) -> ElementaryCondition:
        type = str(java_object.getValueSet().getClass().getName()).split(".")[-1]
        condition: AbstractCondition = {
            "SingletonSetComplement": lambda a, b: self._make_nominal_condition(
                a, b, negated=True
            ),
            "SingletonSet": lambda a, b: self._make_nominal_condition(a, b),
            "Interval": lambda a, b: self._make_numerical_condition(a, b),
        }[type](parent, java_object)
        return condition

    def _make_condition(
        self, parent: CompoundCondition, java_object: Any
    ) -> AbstractCondition:
        type: str = str(java_object.getClass().getName()).split(".")[-1]
        condition: AbstractCondition = {
            "ElementaryCondition": lambda a, b: self._make_elementary_condition(a, b),
            "CompoundCondition": lambda a, b: self._make_compound_condition(a, b),
        }[type](parent, java_object)
        return condition

    def _make_rules(
        self,
        model,
    ) -> list[AbstractRule]:
        rulekit_rules: list = model.model.rules
        output_rules: list[AbstractRule] = [
            self._make_rule(rule) for rule in model.model.rules
        ]
        for i in range(len(rulekit_rules)):
            rulekit_rule = model.model.rules[i]
            output_rules[i].coverage = Coverage(
                rulekit_rule.weighted_p,
                rulekit_rule.weighted_n,
                rulekit_rule.weighted_P,
                rulekit_rule.weighted_N,
            )
            output_rules[i].voting_weight = rulekit_rule.weight
        return output_rules

    @abstractmethod
    def _make_ruleset(self, rules: list[AbstractRule]) -> AbstractRuleSet:
        pass

    @abstractmethod
    def _make_rule(self, rule) -> AbstractRule:
        pass

    @abstractmethod
    def _make_rule_conclusion(self, rule) -> AbstractConclusion:
        pass

    def _make_rule_premise(self, rule) -> AbstractCondition:
        subconditions = rule._java_object.getPremise().getSubconditions()
        premise = CompoundCondition(
            subconditions=[], logic_operator=LogicOperators.CONJUNCTION
        )
        for subcondition in subconditions:
            self._make_condition(premise, subcondition)
        return premise
