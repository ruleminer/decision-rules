from typing import Union

from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import ElementaryCondition
from decision_rules.conditions import LogicOperators
from decision_rules.conditions import NominalCondition
from decision_rules.core.condition import AbstractCondition
from decision_rules.parsers.models import FilterConnector
from decision_rules.parsers.models import FilterInfo
from decision_rules.parsers.models import FilterList
from decision_rules.parsers.models import FilterOperators


class ConditionToFilterParser:
    OPERATOR_MAPPING: dict = {
        # side, is closed, is negated
        ("right", True, True): FilterOperators.greater,
        ("right", True, False): FilterOperators.lower_equal,
        ("right", False, True): FilterOperators.greater_equal,
        ("right", False, False): FilterOperators.lower,
        ("left", True, True): FilterOperators.lower,
        ("left", True, False): FilterOperators.greater_equal,
        ("left", False, True): FilterOperators.lower_equal,
        ("left", False, False): FilterOperators.greater,
    }
    CONNECTOR_MAPPING = {
        LogicOperators.CONJUNCTION: FilterConnector.AND,
        LogicOperators.ALTERNATIVE: FilterConnector.OR,
    }

    def __init__(self, column_names: list[str]):
        self.column_names: list[str] = column_names

    def parse_condition(self, condition: AbstractCondition) -> Union[FilterInfo, FilterList]:
        if isinstance(condition, CompoundCondition):
            parsed_condition = self.parse_compound_condition(condition)
        elif isinstance(condition, ElementaryCondition):
            parsed_condition = self.parse_numeric_condition(condition)
        elif isinstance(condition, NominalCondition):
            parsed_condition = self.parse_nominal_condition(condition)
        else:
            raise ConditionParsingError(
                "Unknown condition type - unable to parse.")
        return parsed_condition

    def parse_compound_condition(self, condition: CompoundCondition) -> FilterList:
        operator: LogicOperators = condition.logic_operator
        connector: FilterConnector = self.CONNECTOR_MAPPING[operator]
        rule_conditions_list: list[FilterInfo] = []
        conditions: list[AbstractCondition] = condition.subconditions
        for condition in conditions:
            parsed_condition = self.parse_condition(condition)
            rule_conditions_list.append(parsed_condition)
        rule_filter_list: FilterList = FilterList(
            connector=connector,
            filters=rule_conditions_list,
        )
        return rule_filter_list

    def parse_nominal_condition(self, condition: NominalCondition) -> FilterInfo:
        attribute = self.column_names[condition.column_index]
        value = condition.value
        is_negated = condition.negated
        operator = FilterOperators.not_equal if is_negated else FilterOperators.equal
        condition_filter = FilterInfo(
            column_name=attribute,
            operator=operator,
            value=value,
        )
        return condition_filter

    def parse_numeric_condition(self, condition: ElementaryCondition) -> FilterList:
        attribute = self.column_names[condition.column_index]
        condition_sides: list[FilterInfo] = []
        if condition.left > float('-inf'):
            left_condition = FilterInfo(
                column_name=attribute,
                operator=self.OPERATOR_MAPPING[(
                    "left", condition.left_closed, condition.negated)],
                value=condition.left,
            )
            condition_sides.append(left_condition)
        if condition.right < float('inf'):
            right_condition: FilterInfo = FilterInfo(
                column_name=attribute,
                operator=self.OPERATOR_MAPPING[(
                    "right", condition.right_closed, condition.negated)],
                value=condition.right,
            )
            condition_sides.append(right_condition)
        condition_filter = FilterList(
            connector=FilterConnector.AND,
            filters=condition_sides,
        )
        return condition_filter


class ConditionParsingError(Exception):
    pass
