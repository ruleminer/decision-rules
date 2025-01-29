import re
from abc import abstractmethod
from typing import Any
from typing import Iterable

import numpy as np
import pandas as pd
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import ElementaryCondition
from decision_rules.conditions import LogicOperators
from decision_rules.conditions import NominalCondition
from decision_rules.core.rule import AbstractConclusion
from decision_rules.core.rule import AbstractRule
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.ruleset_factories._factories.abstract_factory import AbstractFactory


class AbstractTextRuleSetFactory(AbstractFactory):

    def __init__(self) -> None:
        self.column_indices: dict = None
        self.columns_names: list[str] = None
        self.labels_values: Iterable[Any] = None

    def make(
        self,
        model: list,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> AbstractRuleSet:
        self.X = X_train
        self.y = y_train
        self.decision_attribute_name = y_train.name
        self.labels_values = y_train.unique()
        self.columns_names = X_train.columns.tolist()
        self.column_indices = {
            column_name: i for i, column_name in enumerate(self.columns_names)
        }
        error_messages = []
        rules = []
        for rule_str in model:
            try:
                rule = self._make_rule(rule_str)
                rules.append(rule)
            except ValueError as e:
                error_messages.append(
                    f"Error processing rule '{rule_str}': {e}")

        if error_messages:
            raise ValueError(
                "Errors encountered while processing rules:\n" + "\n".join(error_messages))

        y_uniques, y_counts = np.unique(y_train, return_counts=True)

        ruleset = self._make_ruleset(rules)

        ruleset._calculate_P_N(y_uniques, y_counts)

        ruleset.column_names = self.columns_names
        ruleset.decision_attribute = self.decision_attribute_name

        return ruleset

    def _parse_condition_from_string(self, condition_str: str) -> CompoundCondition:
        conditions = condition_str.split(" AND ")
        compound_condition = CompoundCondition(
            subconditions=[], logic_operator=LogicOperators.CONJUNCTION)
        for condition in conditions:
            attr_name, operator, value, value_type = self._parse_single_condition(
                condition)
            if attr_name not in self.columns_names:
                raise ValueError(
                    f"Attribute {attr_name} not found in columns_names")
            column_index = self.columns_names.index(attr_name)
            if value_type == 'numeric':
                if operator in ['<', '<=']:
                    compound_condition.subconditions.append(ElementaryCondition(
                        column_index=column_index,
                        left=float('-inf'),
                        right=float(value),
                        left_closed=False,
                        right_closed=operator == '<='
                    ))
                elif operator in ['>', '>=']:
                    compound_condition.subconditions.append(ElementaryCondition(
                        column_index=column_index,
                        left=float(value),
                        right=float('inf'),
                        left_closed=operator == '>=',
                        right_closed=False
                    ))
            elif value_type == 'range':
                left, right, left_closed, right_closed = self._parse_numerical_values(
                    value)
                compound_condition.subconditions.append(ElementaryCondition(
                    column_index=column_index,
                    left=left,
                    right=right,
                    left_closed=left_closed,
                    right_closed=right_closed
                ))
            elif value_type == 'categorical':
                condition = NominalCondition(
                    column_index=column_index,
                    value=value
                )
                condition.negated = operator == '!='
                compound_condition.subconditions.append(condition)
        return compound_condition

    def _parse_single_condition(self, condition_str: str):
        pattern = r"(.+?)\s*(<=|>=|<|>|!=|=)\s*(\{.*?\}|(?:<|\()\s*[\d.-]+\s*,\s*[\d.-]+\s*(?:>|\))|[\w\s.'\[\]-]+)"

        match = re.match(pattern, condition_str.strip())
        if not match:
            raise ValueError(f"Invalid condition format: {condition_str}")

        attr_name, operator, value = match.groups()
        if value.startswith("{") and value.endswith("}"):
            value = value[1:-1]
            value_type = 'categorical'
        elif operator == '=' and ((value.startswith("<") and value.endswith(">")) or (value.startswith("(") and value.endswith(")")) or (value.startswith("<") and value.endswith(")")) or (value.startswith("(") and value.endswith(">"))):
            value_type = 'range'
        elif operator in ['<', '>', '<=', '>=']:
            if not re.match(r"\s*[\d]+(\.[\d]+)?\s*$", value):
                raise ValueError(
                    f"Expected a numeric value after {operator}, got: {value}")
            value_type = 'numeric'
        else:
            raise ValueError(
                f"Invalid value format for operator {operator}: {value}")

        return attr_name, operator, value, value_type

    def _parse_numerical_values(self, value_str: str):
        left_closed = False
        right_closed = False
        value_str = value_str.strip()

        if value_str.startswith('<'):
            left_closed = True
        if value_str.endswith('>'):
            right_closed = True

        inner_value_str = value_str[1:-1]
        left_str, right_str = inner_value_str.split(',')

        left = float(left_str) if left_str not in (
            '-inf', '') else float('-inf')
        right = float(right_str) if right_str not in (
            'inf', '') else float('inf')

        return left, right, left_closed, right_closed

    def _make_rule_premise(self, rule_str: str) -> CompoundCondition:
        premise_str = rule_str.split(" THEN ")[0][3:]
        compound_condition = self._parse_condition_from_string(premise_str)
        return compound_condition

    @abstractmethod
    def _make_rule(self, rule_str: str) -> AbstractRule:
        pass

    @abstractmethod
    def _make_rule_conclusion(self, decision_attr: str, decision_value: Any) -> AbstractConclusion:
        pass

    @abstractmethod
    def _make_ruleset(self, rules: list[AbstractRule]) -> AbstractRuleSet:
        pass
