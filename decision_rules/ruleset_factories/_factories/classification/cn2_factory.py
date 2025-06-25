from typing import Type

import pandas as pd

from decision_rules.classification import ClassificationConclusion
from decision_rules.classification import ClassificationRule
from decision_rules.classification import ClassificationRuleSet
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import ElementaryCondition
from decision_rules.conditions import NominalCondition
from decision_rules.core.condition import AbstractCondition
from decision_rules.ruleset_factories._factories.abstract_factory import AbstractFactory
from decision_rules.ruleset_factories.utils.abstract_cn2_factory import AbstractOrangeCN2RuleSetFactory
from decision_rules.ruleset_factories.utils.abstract_cn2_factory import check_if_orange_is_installed_and_correct_version


def get_orange_cn2_factory_class() -> Type[AbstractFactory]:
    """
    Returns the OrangeCN2RuleSetFactory class, performing a lazy check
    to see if the Orange library is installed. If Orange is not installed,
    an ImportError is raised.
    """
    check_if_orange_is_installed_and_correct_version()
    # Import from Orange only after confirming that it is installed
    from Orange.classification.rules import (
        CN2Classifier,
        Rule as CN2Rule,
        Selector,
    )

    class OrangeCN2RuleSetFactory(AbstractOrangeCN2RuleSetFactory):
        """
        Factory that can create a ClassificationRuleSet (from decision_rules)
        based on an Orange CN2Classifier model.
        """

        def make(
            self,
            model: CN2Classifier,
            X_train: pd.DataFrame
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
            ruleset.default_conclusion = self._make_rule_conclusion(
                model.rule_list[-1])

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
    # Return the newly defined class
    return OrangeCN2RuleSetFactory
