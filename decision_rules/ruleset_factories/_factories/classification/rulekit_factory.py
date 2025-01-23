# pylint: disable=protected-access
from typing import List, Type

import pandas as pd

from decision_rules.classification.rule import (ClassificationConclusion,
                                                ClassificationRule)
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.ruleset_factories._factories.abstract_factory import \
    AbstractFactory
from decision_rules.ruleset_factories.utils.abstract_rulekit_factory import (  # pylint: disable=import-outside-toplevel
    AbstractRuleKitRuleSetFactory,
    check_if_rulekit_is_installed_and_correct_version)


def get_rulekit_factory_class() -> Type[AbstractFactory]:
    check_if_rulekit_is_installed_and_correct_version()

    from rulekit.classification import \
        RuleClassifier  # pylint: disable=import-outside-toplevel
    from rulekit.rules import \
        BaseRule as RuleKitRule  # pylint: disable=import-outside-toplevel

    class RuleKitRuleSetFactory(AbstractRuleKitRuleSetFactory):
        """Generates classification ruleset from RuleKit RuleClassifier"""

        def make(
            self, model: RuleClassifier, X_train: pd.DataFrame, y_train: pd.Series
        ) -> ClassificationRuleSet:
            ruleset: ClassificationRuleSet = super().make(model, X_train, y_train)
            ruleset.y_values = self.labels_values
            return ruleset

        def _make_ruleset(
            self, rules: List[ClassificationRule]
        ) -> ClassificationRuleSet:
            return ClassificationRuleSet(rules)

        def _make_rule(self, rule: RuleKitRule) -> ClassificationRule:
            return ClassificationRule(
                premise=self._make_rule_premise(rule),
                conclusion=self._make_rule_conclusion(rule),
                column_names=self.columns_names,
            )

        def _make_rule_conclusion(self, rule: RuleKitRule) -> ClassificationConclusion:
            consequence = rule._java_object.getConsequence()
            consequence_mapping = consequence.getValueSet().getMapping()
            decision_value = consequence.getValueSet().getValue()
            if consequence_mapping is not None:
                decision_value = consequence_mapping.get(int(decision_value))
                if decision_value.__class__.__name__ == "java.lang.String":
                    decision_value = str(decision_value)
                else:
                    decision_value = int(decision_value)
            decision_attribute_name = str(consequence.getAttribute())
            return ClassificationConclusion(decision_value, decision_attribute_name)

        def _calculate_P_N(self, model: RuleClassifier, ruleset: ClassificationRuleSet):
            P = {y_value: 0 for y_value in self.labels_values}
            N = {y_value: 0 for y_value in self.labels_values}
            for rule in model.model.rules:
                decision = int(
                    rule._java_object.getConsequence().getValueSet().getValue()
                )
                N[decision] = rule.weighted_N
                P[decision] = rule.weighted_P
            ruleset.train_P = P
            ruleset.train_N = N

    return RuleKitRuleSetFactory
