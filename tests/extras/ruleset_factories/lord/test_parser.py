import os
from unittest import TestCase

from decision_rules.ruleset_factories._parsers import LordParser
from tests.loaders import load_ruleset_factories_resources_path


class TestLordParser(TestCase):
    """
    Test the LordParser class on a sample file, e.g. "credit_LORD.txt".
    We'll compare the parser output to an expected text file (with lines of rules).
    """

    def test_parse_lord_rules(self):
        rules_dir = load_ruleset_factories_resources_path()

        lord_file_path = os.path.join(rules_dir, "credit_LORD.txt")
        with open(lord_file_path, encoding="utf-8") as f:
            lord_rules_lines = f.readlines()

        parsed_rules = LordParser.parse(lord_rules_lines)

        with open(os.path.join(rules_dir, "credit_lord_parser_output.txt"), encoding="utf-8") as f:
            expected_rule_lines = [line.strip() for line in f]

        self.assertEqual(len(parsed_rules), len(expected_rule_lines))

        for i, rule_text in enumerate(parsed_rules):
            self.assertEqual(rule_text, expected_rule_lines[i])
