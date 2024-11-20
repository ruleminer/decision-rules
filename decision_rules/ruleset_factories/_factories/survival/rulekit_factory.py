# pylint: disable=protected-access
from typing import List

import numpy as np
import pandas as pd
from rulekit.rules import BaseRule as RuleKitRule
from rulekit.survival import SurvivalRules

from decision_rules.ruleset_factories.utils.abstract_rulekit_factory import \
    AbstractRuleKitRuleSetFactory
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator
from decision_rules.survival.kaplan_meier import SurvInfo
from decision_rules.survival.rule import SurvivalConclusion
from decision_rules.survival.rule import SurvivalRule
from decision_rules.survival.ruleset import SurvivalRuleSet


class RuleKitRuleSetFactory(AbstractRuleKitRuleSetFactory):
    """Generates survival ruleset from RuleKit SurvivalRules
    """

    def make(
        self,
        model: SurvivalRules,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> SurvivalRuleSet:
        self.labels_values = y_train.unique()
        self.columns_names = X_train.columns.tolist()
        self.column_indices = {
            column_name: i for i, column_name in enumerate(self.columns_names)
        }

        ruleset = self._make_ruleset([
            self._make_rule(rule, model.survival_time_attr)
            for rule in model.model.rules
        ], survival_time_attr=model.survival_time_attr)

        ruleset.column_names = self.columns_names
        ruleset.decision_attribute = y_train.name

        ruleset.update(
            X_train, y_train
        )

        return ruleset

    def _make_ruleset(self, rules: List[SurvivalRule], survival_time_attr: str) -> SurvivalRuleSet:
        return SurvivalRuleSet(rules, survival_time_attr)

    def _make_rule(self, rule: RuleKitRule, survival_time_attr: str) -> SurvivalRule:
        rule = SurvivalRule(
            premise=self._make_rule_premise(rule),
            conclusion=self._make_rule_conclusion(rule),
            column_names=self.columns_names,
            survival_time_attr=survival_time_attr
        )
        return rule

    def _make_rule_conclusion(self, rule: RuleKitRule) -> SurvivalConclusion:
        consequence = rule._java_object.getConsequence()

        estimator = self._get_estimator(rule._java_object.getEstimator())
        decision_attribute_name = str(consequence.getAttribute())
        conclusion = SurvivalConclusion(
            value=None, column_name=decision_attribute_name, fixed=True)
        conclusion.estimator = estimator
        return conclusion

    def _get_estimator(self, java_estimator) -> KaplanMeierEstimator:
        java_times = java_estimator.getTimes()

        times = []
        events_count = []
        at_risk_count = []
        probabilities = []

        for time in java_times:
            times.append(float(time))
            events_count.append(int(java_estimator.getEventsCountAt(time)))
            at_risk_count.append(int(java_estimator.getRiskSetCountAt(time)))
            probabilities.append(float(java_estimator.getProbabilityAt(time)))

        times = np.array(times)
        events_count = np.array(events_count)
        at_risk_count = np.array(at_risk_count)
        probabilities = np.array(probabilities)

        surv_info = SurvInfo(
            time=times,
            events_count=events_count,
            censored_count=np.zeros(len(java_times)),
            at_risk_count=at_risk_count,
            probability=probabilities
        )
        return KaplanMeierEstimator(surv_info)

    def _calculate_P_N(
        self,
        model: SurvivalRules,
        ruleset: SurvivalRuleSet
    ):
        pass
