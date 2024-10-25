import json
import os
import unittest

from decision_rules.classification.rule import ClassificationRule
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.parsers import RuleToFilterParser
from decision_rules.serialization import JSONSerializer
from tests.loaders import load_resources_path


class TestRulesetParser(unittest.TestCase):

    def _prepare_test_rule(self) -> ClassificationRule:
        ruleset_file_path: str = os.path.join(
            load_resources_path(), 'iris_ruleset.json')
        with open(ruleset_file_path, 'r', encoding='utf-8') as file:
            return JSONSerializer.deserialize(
                json.load(file),
                ClassificationRuleSet
            )

    def test_parser(self):
        ruleset = self._prepare_test_rule()
        parser = RuleToFilterParser(ruleset)
        filters = parser.parse_ruleset_to_filters()
        self.assertTrue(True)
