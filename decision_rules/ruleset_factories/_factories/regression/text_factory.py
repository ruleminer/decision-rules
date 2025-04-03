import re
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
from decision_rules.helpers import get_measure_function_by_name
from decision_rules.regression.rule import RegressionConclusion
from decision_rules.regression.rule import RegressionRule
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.ruleset_factories.utils.abstract_text_factory import AbstractTextRuleSetFactory
from decision_rules.core.exceptions import InvalidMeasureNameException
from decision_rules.core.exceptions import RuleConclusionFormatException
from decision_rules.core.exceptions import DecisionAttributeMismatchException
from decision_rules.core.exceptions import RuleConclusionFloatConversionException


class TextRuleSetFactory(AbstractTextRuleSetFactory):
    """Generates regression ruleset from list of str rules
    """

    def make(
        self,
        model: list,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        measure_name: Union[str, Callable] = "C2"
    ) -> RegressionRuleSet:

        if isinstance(measure_name, str):
            measure = get_measure_function_by_name(measure_name)
        elif callable(measure_name):
            measure = measure_name
        else:
            raise InvalidMeasureNameException()

        ruleset: RegressionRuleSet = super().make(
            model, X_train, y_train
        )
        ruleset.y_values = self.labels_values

        ruleset.update(
            X_train, y_train,
            measure=measure
        )
        return ruleset

    def _make_ruleset(self, rules: List[RegressionRule]) -> RegressionRuleSet:
        return RegressionRuleSet(rules)

    def _make_rule(self, rule_str: str) -> RegressionRule:
        premise = self._make_rule_premise(rule_str)
        conclusion = self._make_rule_conclusion(rule_str)
        if conclusion is None:
            conclusion = self._calculate_conclusion_automatically()
        return RegressionRule(premise=premise, conclusion=conclusion, column_names=self.columns_names)

    def _make_rule_conclusion(self, rule_str: str) -> Optional[RegressionConclusion]:
        rule_parts = rule_str.split(" THEN ", 1)
        if len(rule_parts) < 2 or not rule_parts[1].strip():
            return None

        _, conclusion_part = rule_parts

        pattern = r"\s*(.+?)\s*=\s*\{(.*?)\}\s*(?:\[(.*?),\s*(.*?)\])?"
        match = re.search(pattern, conclusion_part)

        if not match:
            raise RuleConclusionFormatException(conclusion_part=conclusion_part)

        decision_attribute_name, decision_value, low_value, high_value = match.groups()

        if decision_attribute_name != self.decision_attribute_name:
            raise DecisionAttributeMismatchException(
                given_attribute=decision_attribute_name,
                expected_attribute=self.decision_attribute_name
            )

        try:
            decision_value = float(decision_value)
            low_value = float(low_value) if low_value else decision_value
            high_value = float(high_value) if high_value else decision_value
        except ValueError as e:
            raise RuleConclusionFloatConversionException()

        return RegressionConclusion(value=decision_value, column_name=decision_attribute_name, fixed=True, low=low_value, high=high_value)

    def _calculate_conclusion_automatically(self) -> RegressionConclusion:

        return RegressionConclusion(value=None, column_name=self.decision_attribute_name)
