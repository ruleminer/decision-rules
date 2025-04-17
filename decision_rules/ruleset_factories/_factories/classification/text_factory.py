# pylint: disable=protected-access
import re
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from decision_rules.classification.rule import ClassificationConclusion
from decision_rules.classification.rule import ClassificationRule
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.helpers import get_measure_function_by_name
from decision_rules.ruleset_factories.utils.abstract_text_factory import AbstractTextRuleSetFactory
from decision_rules.core.exceptions import InvalidMeasureNameException
from decision_rules.core.exceptions import RuleConclusionFormatException
from decision_rules.core.exceptions import DecisionAttributeMismatchException


class TextRuleSetFactory(AbstractTextRuleSetFactory):
    """Generates classification ruleset from list of str rules
    """

    def make(
        self,
        model: list,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        measure_name: Union[str, Callable] = "C2"
    ) -> ClassificationRuleSet:

        if isinstance(measure_name, str):
            measure = get_measure_function_by_name(measure_name)
        elif callable(measure_name):
            measure = measure_name
        else:
            raise InvalidMeasureNameException()

        ruleset: ClassificationRuleSet = super().make(
            model, X_train, y_train
        )
        ruleset.y_values = self.labels_values
        ruleset.update(
            X_train, y_train,
            measure=measure
        )
        return ruleset

    def _make_ruleset(self, rules: List[ClassificationRule]) -> ClassificationRuleSet:
        return ClassificationRuleSet(rules)

    def _make_rule(self, rule_str: str) -> ClassificationRule:
        premise = self._make_rule_premise(rule_str)
        conclusion = self._make_rule_conclusion(rule_str)
        if conclusion is None:
            conclusion = self._calculate_conclusion_automatically(premise)

        return ClassificationRule(premise=premise, conclusion=conclusion, column_names=self.columns_names)

    def _make_rule_conclusion(self, rule_str: str) -> Optional[ClassificationConclusion]:
        rule_parts = rule_str.split(" THEN ", 1)
        if len(rule_parts) < 2 or not rule_parts[1].strip():
            return None

        _, conclusion_part = rule_parts

        pattern = r"\s*(.+?)\s*=\s*\{(.*?)\}\s*"
        match = re.search(pattern, conclusion_part)

        if not match:
            raise RuleConclusionFormatException(conclusion_part=conclusion_part)

        decision_attribute_name, decision_value = match.groups()

        if decision_attribute_name != self.decision_attribute_name:
            raise DecisionAttributeMismatchException(
                given_attribute=decision_attribute_name,
                expected_attribute=self.decision_attribute_name
            )

        return ClassificationConclusion(column_name=decision_attribute_name, value=decision_value)

    def _calculate_conclusion_automatically(self, premise):
        covered_mask = premise._calculate_covered_mask(self.X.values)
        covered_y = self.y[covered_mask]

        y_uniques, y_counts = np.unique(covered_y, return_counts=True)

        max_counts_indices = np.where(y_counts == np.max(y_counts))[0]

        if len(max_counts_indices) > 1:
            overall_counts = np.array([self.y.value_counts().get(
                class_name) for class_name in y_uniques[max_counts_indices]])
            chosen_class = y_uniques[max_counts_indices][np.argmin(
                overall_counts)]
        else:
            chosen_class = y_uniques[max_counts_indices[0]]

        return ClassificationConclusion(column_name=self.decision_attribute_name, value=chosen_class)
