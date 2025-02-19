import pandas as pd
import unittest
from decision_rules.classification import ClassificationRule
from decision_rules.classification import ClassificationRuleSet
from decision_rules.classification import ClassificationConclusion
from decision_rules.conditions import ElementaryCondition, CompoundCondition
from decision_rules import measures

class TestCalculateRulesetStats(unittest.TestCase):
    def test_single_elementary(self):
        """
        Test a rule with a single elementary condition.
        Expect total_conditions_count = 1 and avg_conditions_count = 1.
        """
        elementary = ElementaryCondition(column_index=0, left=10)
        rule = ClassificationRule(
            premise=elementary,
            conclusion=ClassificationConclusion(value='0', column_name='label'),
            column_names=['a']
        )
        ruleset = ClassificationRuleSet([rule])
        df = pd.DataFrame({'a': range(10)})
        series = pd.Series(['0'] * 10)
        
        ruleset.update(df, series, measure=measures.c2)
        stats = ruleset.calculate_ruleset_stats()
        
        assert stats['total_conditions_count'] == 1
        assert stats['avg_conditions_count'] == 1

    def test_compound_condition(self):
        """
        Test a rule whose premise is a CompoundCondition with two elementary conditions.
        Expect total_conditions_count = 2 and avg_conditions_count = 2.
        """
        compound = CompoundCondition(
            logic_operator='AND',
            subconditions=[
                ElementaryCondition(column_index=0, left=18),
                ElementaryCondition(column_index=0, left=20)
            ]
        )
        rule = ClassificationRule(
            premise=compound,
            conclusion=ClassificationConclusion(value='1', column_name='label'),
            column_names=['a']
        )
        ruleset = ClassificationRuleSet([rule])
        df = pd.DataFrame({'a': list(range(25))})
        series = pd.Series(['1'] * 25)
        
        ruleset.update(df, series, measure=measures.c2)
        stats = ruleset.calculate_ruleset_stats()
        
        assert stats['total_conditions_count'] == 2
        assert stats['avg_conditions_count'] == 2

    def test_nested_compound_condition(self):
        """
        Test a rule where the main CompoundCondition consists of:
          - one elementary condition,
          - and a nested CompoundCondition (logic_operator OR) with two elementary conditions.
        Expect total_conditions_count = 3 and avg_conditions_count = 3.
        """
        inner_compound = CompoundCondition(
            logic_operator='OR',
            subconditions=[
                ElementaryCondition(column_index=0, left=5),
                ElementaryCondition(column_index=0, left=7)
            ]
        )
        compound = CompoundCondition(
            logic_operator='AND',
            subconditions=[
                ElementaryCondition(column_index=0, left=3),
                inner_compound
            ]
        )
        rule = ClassificationRule(
            premise=compound,
            conclusion=ClassificationConclusion(value='1', column_name='label'),
            column_names=['a']
        )
        ruleset = ClassificationRuleSet([rule])
        df = pd.DataFrame({'a': list(range(15))})
        series = pd.Series(['1'] * 15)
        
        ruleset.update(df, series, measure=measures.c2)
        stats = ruleset.calculate_ruleset_stats()
        
        assert stats['total_conditions_count'] == 3
        assert stats['avg_conditions_count'] == 3

    def test_multiple_rules(self):
        """
        Test a ruleset with two rules:
          - Rule 1: a single elementary condition (1 condition).
          - Rule 2: a CompoundCondition with two elementary conditions (2 conditions).
        Expect total_conditions_count = 3, avg_conditions_count = 1.5, and rules_count = 2.
        """
        rule1 = ClassificationRule(
            premise=ElementaryCondition(column_index=0, left=10),
            conclusion=ClassificationConclusion(value='0', column_name='label'),
            column_names=['a']
        )
        compound = CompoundCondition(
            logic_operator='AND',
            subconditions=[
                ElementaryCondition(column_index=0, left=15),
                ElementaryCondition(column_index=0, left=20)
            ]
        )
        rule2 = ClassificationRule(
            premise=compound,
            conclusion=ClassificationConclusion(value='0', column_name='label'),
            column_names=['a']
        )
        ruleset = ClassificationRuleSet([rule1, rule2])
        df = pd.DataFrame({'a': list(range(20))})
        series = pd.Series(['0'] * 20)
        
        ruleset.update(df, series, measure=measures.c2)
        stats = ruleset.calculate_ruleset_stats()
        
        assert stats['rules_count'] == 2
        assert stats['total_conditions_count'] == 3
        assert stats['avg_conditions_count'] == 1.5