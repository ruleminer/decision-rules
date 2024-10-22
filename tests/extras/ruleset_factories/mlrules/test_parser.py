# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os
from unittest import TestCase

from decision_rules.ruleset_factories._parsers import MLRulesParser

from tests.loaders import load_ruleset_factories_resources_path


class TestMLRulesParser(TestCase):
    """
    Test the MLRulesParser class. The test is performed on an example from "credit" dataset.
    """

    def test_parse_ml_rules(self):
        rules_dir = load_ruleset_factories_resources_path()
        with open(os.path.join(rules_dir, "credit_MLRules.txt")) as file:
            ml_rules_lines = file.readlines()
        parsed_rules = MLRulesParser.parse(ml_rules_lines)
        with open(os.path.join(rules_dir, "credit_parser_output.txt")) as file:
            check_rules = file.readlines()
        check_rules = [rule.strip() for rule in check_rules]
        self.assertEqual(parsed_rules, check_rules)
