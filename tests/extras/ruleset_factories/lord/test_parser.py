import os
from unittest import TestCase

from decision_rules.ruleset_factories._parsers import LordParser
from tests.loaders import load_ruleset_factories_resources_path


class TestLordParser(TestCase):
    """
    Test the LordParser class on a sample file, e.g. "credit_LORD.txt".
    We'll compare the parser output to an expected text file (with lines of rules)
    and optionally check the extracted heuristic_value as well.
    """

    def test_parse_lord_rules(self):
        rules_dir = load_ruleset_factories_resources_path()

        lord_file_path = os.path.join(rules_dir, "credit_LORD.txt")
        with open(lord_file_path, encoding="utf-8") as f:
            lord_rules_lines = f.readlines()

        parsed_tuples = LordParser.parse(lord_rules_lines)

        with open(os.path.join(rules_dir, "credit_lord_parser_output.txt"), encoding="utf-8") as f:
            expected_rule_lines = [line.strip() for line in f]

        self.assertEqual(len(parsed_tuples), len(expected_rule_lines))

        for i, (rule_text, hv) in enumerate(parsed_tuples):
            self.assertEqual(rule_text, expected_rule_lines[i])

        heur_path = os.path.join(rules_dir, "credit_lord_parser_heuristics")
        if os.path.exists(heur_path):
            with open(heur_path, encoding="utf-8") as f:
                expected_heuristics = [float(line.strip()) for line in f]
            self.assertEqual(len(parsed_tuples), len(expected_heuristics))
            for i, (rule_text, hv) in enumerate(parsed_tuples):
                self.assertAlmostEqual(hv, expected_heuristics[i], places=6)
