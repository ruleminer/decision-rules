from typing import Callable
from typing import Union

import pandas as pd

from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.ruleset_factories._factories.classification.text_factory import TextRuleSetFactory
from decision_rules.ruleset_factories._parsers import LordParser
from decision_rules.ruleset_factories.utils.abstract_lord_factory import AbstractLordRuleSetFactory


class LordRuleSetFactory(AbstractLordRuleSetFactory):
    """
    Factory for creating a ClassificationRuleSet from a list of lines of LORD ruleset.

    Information about the LORD algorithm and format can be found at:
    https://github.com/vqphuynh/LORD

    """

    def make(
        self,
        model: list[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        measure_name: Union[str, Callable] = "C2",
    ) -> ClassificationRuleSet:
        """

        Args:
            model: List of rules extracted from the "Rule set" section of the LORD output file.
                   Each rule is represented as a single line string
            X_train: pandas dataframe with features
            y_train: data series with dependent variable
            measure_name: voting measure used to calculate rule voting weights

        Returns:
            ClassificationRuleSet: a set of classification
        """
        parsed_tuples: list[tuple[str, float]] = LordParser.parse(model)

        rule_texts: list[str] = [tpl[0] for tpl in parsed_tuples]
        heuristic_values: list[float] = [tpl[1] for tpl in parsed_tuples]
        ruleset: ClassificationRuleSet = TextRuleSetFactory().make(
            rule_texts, X_train, y_train, measure_name=measure_name
        )
        for rule_obj, hv in zip(ruleset.rules, heuristic_values):
            rule_obj.voting_weight = hv

        return ruleset
