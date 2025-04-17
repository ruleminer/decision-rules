import re
from typing import Any, Iterable, List

import numpy as np
import pandas as pd
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator
from decision_rules.survival.rule import SurvivalConclusion
from decision_rules.survival.rule import SurvivalRule
from decision_rules.survival.ruleset import SurvivalRuleSet
from decision_rules.ruleset_factories.utils.abstract_text_factory import AbstractTextRuleSetFactory
from decision_rules.core.exceptions import InvalidSurvivalTimeAttributeException
from decision_rules.core.exceptions import RuleConclusionFormatException
from decision_rules.core.exceptions import RuleConclusionFloatConversionException

class TextRuleSetFactory(AbstractTextRuleSetFactory):
    """Generates survival ruleset from list of str rules
    """

    def make(
        self,
        model: list,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        survival_time_attr: str = "survival_time",
    ) -> SurvivalRuleSet:
        if survival_time_attr not in X_train.columns.tolist():
            raise InvalidSurvivalTimeAttributeException(attribute=survival_time_attr)
        labels_values, y_counts = np.unique(y_train, return_counts=True)
        ruleset: SurvivalRuleSet = self._build_ruleset(
            model=model,
            y_counts=y_counts,
            decision_attribute_name=y_train.name,
            labels_values = labels_values,
            columns_names=X_train.columns.tolist(),
            survival_time_attr = survival_time_attr
        )
        ruleset.y_values = labels_values
        ruleset.update(X_train, y_train)
        return ruleset
    
    def _build_ruleset(
        self,
        model: list,
        y_counts: np.ndarray,
        decision_attribute_name: str,
        labels_values: Iterable[Any],
        columns_names: list[str],
        survival_time_attr: str
    ) -> SurvivalRuleSet:
        self.survival_time_attr = survival_time_attr
        ruleset = super()._build_ruleset(
            model,
            y_counts,
            decision_attribute_name,
            labels_values,
            columns_names
        )
        return ruleset

    def _make_ruleset(self, rules: List[SurvivalRule]) -> SurvivalRuleSet:
        return SurvivalRuleSet(rules, self.survival_time_attr)

    def _make_rule(self, rule_str: str) -> SurvivalRule:
        premise = self._make_rule_premise(rule_str)
        conclusion = self._make_rule_conclusion(rule_str)
        return SurvivalRule(premise=premise, conclusion=conclusion, column_names=self.columns_names, survival_time_attr=self.survival_time_attr)

    def _make_rule_conclusion(self, rule_str: str):
        rule_parts = rule_str.split(" THEN ", 1)
        if len(rule_parts) < 2 or not rule_parts[1].strip():
            return SurvivalConclusion(value=None, column_name=self.decision_attribute_name)

        _, conclusion_part = rule_parts
        pattern = r"probabilities\s*=\s*\[(.*?)\],?\s*times\s*=\s*\[(.*?)\]"
        match = re.search(pattern, conclusion_part)

        if not match:
            raise RuleConclusionFormatException(conclusion_part=conclusion_part)

        try:
            probabilities_str, times_str = match.groups()
            probabilities = [float(prob)
                             for prob in probabilities_str.split(',')]
            times = [float(time) for time in times_str.split(',')]
        except ValueError as e:
            raise RuleConclusionFloatConversionException()

        zeros = [0] * len(times)

        kaplan_meier_estimator = KaplanMeierEstimator().update({
            "times": times,
            "probabilities": probabilities,
            "events_count": zeros,
            "censored_count": zeros,
            "at_risk_count": zeros,
        }, update_additional_indicators=True)
        conclusion = SurvivalConclusion(
            value=None, column_name=self.decision_attribute_name, fixed=True)
        conclusion.estimator = kaplan_meier_estimator
        return conclusion
