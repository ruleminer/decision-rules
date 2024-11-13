# pylint: disable=protected-access
from typing import List

from pandas import DataFrame
from pandas import Series
from rulekit.regression import RuleRegressor
from rulekit.rules import BaseRule as RuleKitRule

from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.regression.rule import RegressionConclusion
from decision_rules.regression.rule import RegressionRule
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.ruleset_factories.utils.abstract_rulekit_factory import AbstractRuleKitRuleSetFactory


class RuleKitRuleSetFactory(AbstractRuleKitRuleSetFactory):
    """Generates regression ruleset from RuleKit RuleRegressor
    """

    def _make_ruleset(self, rules: List[RegressionRule]) -> RegressionRuleSet:
        return RegressionRuleSet(rules)

    def _make_rule(self, rule: RuleKitRule) -> RegressionRule:
        return RegressionRule(
            premise=self._make_rule_premise(rule),
            conclusion=self._make_rule_conclusion(rule),
            column_names=self.columns_names
        )

    def _make_rule_conclusion(self, rule: RuleKitRule) -> RegressionConclusion:
        consequence = rule._java_object.getConsequence()
        decision_value = consequence.getValueSet().getValue()
        decision_attribute_name = str(consequence.getAttribute())
        return RegressionConclusion(decision_value, decision_attribute_name)

    def _calculate_P_N(
        self,
        model: RuleRegressor,
        ruleset: RegressionRuleSet
    ):
        pass

    def make(self, model: RuleRegressor, X_train: DataFrame, y_train: Series) -> AbstractRuleSet:
        if not model.get_params().get('mean_based_regression', False):
            raise NotImplementedError(
                'ruleset_factories package does not support median based regression any more.' +
                'use "mean_based_regression" parameter of RuleRegressor class to enable mean ' +
                'based regression.'
            )
        return super().make(model, X_train, y_train)
