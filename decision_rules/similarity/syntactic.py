from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import ElementaryCondition
from decision_rules.conditions import NominalCondition
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.core.simplifier import RulesetSimplifier


class SyntacticRuleSimilarityCalculator:
    """
    Calculator of syntactic rule similarity.
    Caveat: the assumption is that the conditions in rules are connected only with conjunction operators.
    """

    def __init__(self, ruleset1: AbstractRuleSet, ruleset2: AbstractRuleSet, dataset: pd.DataFrame):
        self.ruleset1 = RulesetSimplifier(ruleset1).simplify()
        self.ruleset2 = RulesetSimplifier(ruleset2).simplify()
        self.dataset = dataset
        self.ruleset1 = self._parse_rules_to_conditions(self.ruleset1)
        self.ruleset2 = self._parse_rules_to_conditions(self.ruleset2)

    def calculate(self) -> np.ndarray:
        # calculate rule similarity in a matrix of rule pairs in a vectorized way
        rule_pairs = np.array(list(product(self.ruleset1, self.ruleset2)))
        result = [self._calculate_rule_sim(*rule_pair)
                  for rule_pair in rule_pairs]
        result = np.array(result)
        result = result.reshape(len(self.ruleset1), len(self.ruleset2))
        return result

    def _calculate_rule_sim(self, rule1: dict, rule2: dict) -> float:
        # calculations for denominator
        denominator = len(rule1["elementary"]) + len(rule2["elementary"]) + \
            len(rule1["nominal"]) + len(rule2["nominal"])
        # elementary conditions sums
        elem_sum = self._calculate_elementary_condition_sim_sum(
            rule1["elementary"], rule2["elementary"])
        # nominal conditions sums
        nomin_sum = self._calculate_nominal_condition_sim_sum(
            rule1["nominal"], rule2["nominal"])
        return (elem_sum + nomin_sum) / denominator

    def _parse_rules_to_conditions(self, ruleset: AbstractRuleSet) -> list[dict]:
        rules = []
        for rule in ruleset.rules:
            rule_conditions = defaultdict(dict)
            all_conditions = rule.premise.subconditions if isinstance(
                rule.premise, CompoundCondition) else [rule.premise]
            for condition in all_conditions:
                if isinstance(condition, ElementaryCondition):
                    key = ruleset.column_names[condition.column_index]
                    left = self._evaluate_boundary(
                        condition.left, key)
                    right = self._evaluate_boundary(
                        condition.right, key)
                    rule_conditions["elementary"][key] = left, right
                elif isinstance(condition, NominalCondition):
                    key = ruleset.column_names[condition.column_index]
                    rule_conditions["nominal"][key] = rule_conditions["nominal"].get(
                        key, []) + [condition.value]
                else:
                    raise NotImplementedError(
                        "Only elementary and nominal conditions are supported")
            rules.append(rule_conditions)
        return rules

    def _evaluate_boundary(self, boundary: float, column_name: str) -> float:
        # if a condition is one-sided, change the appropriate +/- inf bound
        # to the actual max/min value of the column in the dataset
        if boundary == float("inf"):
            return self.dataset[column_name].max()
        if boundary == float("-inf"):
            return self.dataset[column_name].min()
        return boundary

    def _calculate_elementary_condition_sim_sum(self, elementary_conditions1, elementary_conditions2) -> float:
        keys = set(elementary_conditions1) & set(elementary_conditions2)
        sim_sum = 0.0
        for key in keys:
            interval1, interval2 = elementary_conditions1[key], elementary_conditions2[key]
            overlap = self._calculate_overlap(interval1, interval2)
            rirj = overlap / (interval1[1] - interval1[0])
            sim_sum += rirj
            rjri = overlap / (interval2[1] - interval2[0])
            sim_sum += rjri
        return sim_sum

    @staticmethod
    def _calculate_nominal_condition_sim_sum(nominal_conditions1, nominal_conditions2) -> float:
        keys = set(nominal_conditions1) & set(nominal_conditions2)
        sim_sum = 0.0
        for key in keys:
            overlap = set(nominal_conditions1[key]) & set(
                nominal_conditions2[key])
            set_ij = set(nominal_conditions1[key])
            set_ji = set(nominal_conditions2[key])
            sim_sum += len(overlap) / len(set_ij)
            sim_sum += len(overlap) / len(set_ji)
        return sim_sum

    @staticmethod
    def _calculate_overlap(interval1: tuple[float, float], interval2: tuple[float, float]) -> float:
        # helper function to calculate overlap between two intervals
        left_bound = max(interval1[0], interval2[0])
        right_bound = min(interval1[1], interval2[1])
        return max(0.0, right_bound - left_bound)
