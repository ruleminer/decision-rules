import copy

from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import ElementaryCondition
from decision_rules.conditions import LogicOperators
from decision_rules.core.condition import AbstractCondition
from decision_rules.core.rule import AbstractRule
from decision_rules.core.ruleset import AbstractRuleSet


class RulesetSimplifier:
    def __init__(self, ruleset: AbstractRuleSet):
        self.ruleset = copy.deepcopy(ruleset)

    def simplify(self) -> AbstractRuleSet:
        rules: list[AbstractRule] = self.ruleset.rules
        for rule in rules:
            self._simplify_rule(rule)
        return self.ruleset

    @staticmethod
    def _simplify_rule(rule: AbstractRule):
        if isinstance(rule.premise, CompoundCondition):
            rule.premise = RulesetSimplifier._simplify_compound_condition(
                rule.premise)

    @staticmethod
    def _simplify_compound_condition(condition: CompoundCondition) -> CompoundCondition:
        if condition.logic_operator != LogicOperators.CONJUNCTION:
            raise NotImplementedError(
                "Currently only compound conditions with conjunction are supported.")
        new_subconditions: dict[int, AbstractCondition] = {}
        for subcondition in condition.subconditions:
            attr = subcondition.column_index
            if attr not in new_subconditions:
                new_subconditions[attr] = subcondition
                continue
            if isinstance(subcondition, ElementaryCondition):
                left, right = subcondition.left, subcondition.right
                if left > new_subconditions[attr].left:
                    new_subconditions[attr].left = left
                    new_subconditions[attr].left_closed = subcondition.left_closed
                if right < new_subconditions[attr].right:
                    new_subconditions[attr].right = right
                    new_subconditions[attr].right_closed = subcondition.right_closed
        new_subconditions: list[AbstractCondition] = list(
            new_subconditions.values())
        return CompoundCondition(new_subconditions)
