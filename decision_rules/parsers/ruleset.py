from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.parsers.condition import ConditionToFilterParser
from decision_rules.parsers.models import FilterConnector
from decision_rules.parsers.models import FilterList


class RuleToFilterParser:
    """
    Class for converting ruleset JSON/dictionary to filtering info.
    """

    def __init__(self, ruleset: AbstractRuleSet, connector: FilterConnector = FilterConnector.OR):
        self.ruleset = ruleset
        self.connector = connector
        self.parsed_rules_ids: list[str] = []
        self.condition_parser = ConditionToFilterParser(
            column_names=ruleset.column_names)

    def parse_ruleset_to_filters(self) -> FilterList:
        rules_list: list[FilterList] = []
        for raw_rule in self.ruleset.rules:
            self.parsed_rules_ids.append(raw_rule.uuid)
            parsed_rule: FilterList = self.condition_parser.parse_condition(
                raw_rule.premise)
            rules_list.append(parsed_rule)
        ruleset_filter_list: FilterList = FilterList(
            connector=self.connector,
            filters=rules_list
        )
        return ruleset_filter_list
