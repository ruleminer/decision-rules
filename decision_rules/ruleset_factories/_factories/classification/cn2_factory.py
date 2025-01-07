import pandas as pd
from decision_rules.classification import (
    ClassificationConclusion,
    ClassificationRule,
    ClassificationRuleSet,
)
from decision_rules.core.condition import AbstractCondition
from decision_rules.conditions import (
    CompoundCondition,
    ElementaryCondition,
    NominalCondition,
)
from Orange.classification.rules import CN2Classifier
from Orange.classification.rules import Rule as CN2Rule
from Orange.classification.rules import Selector


class OrangeCN2RuleSetFactory:

    def make(
        self, model: CN2Classifier, X_train: pd.DataFrame
    ) -> ClassificationRuleSet:
        ruleset: ClassificationRuleSet = ClassificationRuleSet(
            rules=[
                self._make_rule(
                    rule,
                    column_names=list(X_train.columns),
                )
                for rule in model.rule_list[:-1]
            ]
        )
        # last rule is a default one
        ruleset.default_conclusion = self._make_rule_conclusion(model.rule_list[-1])

        return ruleset

    def _make_rule(
        self, cn2_rule: CN2Rule, column_names: list[str]
    ) -> ClassificationRule:
        return ClassificationRule(
            premise=self._make_rule_premise(cn2_rule),
            conclusion=self._make_rule_conclusion(cn2_rule),
            column_names=column_names,
        )

    def _make_rule_premise(self, cn2_rule: CN2Rule) -> CompoundCondition:
        return CompoundCondition(
            subconditions=[
                self._make_subcondition(selector) for selector in cn2_rule.selectors
            ]
        )

    def _make_subcondition(self, selector: Selector) -> AbstractCondition:
        # tiny wrapping function to return negated version of given condition
        def negated(condition: AbstractCondition) -> AbstractCondition:
            condition.negated = not condition.negated
            return condition

        # maps different selectors types for decision-rules conditions
        return {
            "==": lambda c_index, c_value: NominalCondition(
                column_index=c_index, value=c_value
            ),
            "!=": lambda c_index, c_value: negated(
                NominalCondition(column_index=c_index, value=c_value)
            ),
            "<=": lambda c_index, c_value: ElementaryCondition(
                column_index=c_index, right=float(c_value), right_closed=True
            ),
            ">=": lambda c_index, c_value: ElementaryCondition(
                column_index=c_index, left=float(c_value), left_closed=True
            ),
        }[selector.op](selector.column, selector.value)

    def _make_rule_conclusion(self, cn2_rule: CN2Rule) -> ClassificationConclusion:
        return ClassificationConclusion(
            column_name=cn2_rule.domain.class_var.name, value=cn2_rule.prediction
        )