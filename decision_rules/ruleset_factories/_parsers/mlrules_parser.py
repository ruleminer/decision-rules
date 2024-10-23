import re


class MLRulesParser:
    BOOLEAN_CONDITION_PATTERN = re.compile("(?P<column>.*) is (?P<value>.*)$")
    CATEGORICAL_CONDITION_PATTERN = re.compile(
        "(?P<column>.*)=(?P<value>.*) is (?P<equality>.*)$")
    NUMERICAL_CONDITION_PATTERN = re.compile(
        "(?P<column>.*) (?P<operator>[<>=]{1,2}) (?P<value>.*)$")
    RANGE_CONDITION_PATTERN = re.compile(
        "(?P<column>.*) in \[(?P<min>[.0-9]*),(?P<max>[.0-9]*)]$")
    CONCLUSION_PATTERN = re.compile(
        "=> vote for (?P<class_attribute>.*) (?P<class>.*) with weight [\.0-9]*$")

    @staticmethod
    def parse(rules: list[str]) -> list[str]:
        processed_rules = []

        # iterate over lines
        new_rule = ""
        for rule_line in rules:

            # skip empty lines
            if not rule_line.strip():
                continue

            # identify the beginning of a rule
            if "Rule: " in rule_line:
                new_rule = "IF "
                continue

            # add categorical condition
            match = MLRulesParser.CATEGORICAL_CONDITION_PATTERN.match(
                rule_line)
            if match:
                new_rule += MLRulesParser._parse_categorical_condition(match)
                continue

            # add boolean condition
            match = MLRulesParser.BOOLEAN_CONDITION_PATTERN.match(rule_line)
            if match:
                new_rule += MLRulesParser._parse_boolean_condition(match)
                continue

            # add numerical condition
            match = MLRulesParser.NUMERICAL_CONDITION_PATTERN.match(rule_line)
            if match:
                new_rule += MLRulesParser._parse_numerical_condition(match)
                continue

            # add range condition
            match = MLRulesParser.RANGE_CONDITION_PATTERN.match(rule_line)
            if match:
                new_rule += MLRulesParser._parse_range_condition(match)
                continue

            # add conclusion
            match = MLRulesParser.CONCLUSION_PATTERN.match(rule_line)
            if match:
                new_rule = new_rule[:-5]
                new_rule += MLRulesParser._parse_conclusion(match)
                processed_rules.append(new_rule)
                new_rule = ""
                continue

        return processed_rules

    @staticmethod
    def _parse_boolean_condition(match: re.Match) -> str:
        return match["column"].strip() + " = " + "{" + match["value"] + "} AND "

    @staticmethod
    def _parse_categorical_condition(match: re.Match) -> str:
        column = match["column"].strip()
        operator = " = " if match["equality"] == "t" else " != "
        return column + operator + "{" + match["value"] + "} AND "

    @staticmethod
    def _parse_numerical_condition(match: re.Match) -> str:
        return match["column"].strip() + " " + match["operator"] + " " + match["value"] + " AND "

    @staticmethod
    def _parse_range_condition(match: re.Match) -> str:
        column = match["column"].strip()
        return (column + " >= " + match["min"] + " AND "
                + column + " <= " + match["max"] + " AND ")

    @staticmethod
    def _parse_conclusion(match: re.Match) -> str:
        return f" THEN {match['class_attribute']} = " + "{" + match["class"] + "}"
