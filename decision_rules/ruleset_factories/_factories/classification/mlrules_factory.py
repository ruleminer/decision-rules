from typing import Callable
from typing import Union

import pandas as pd
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.ruleset_factories._factories.classification.text_factory import TextRuleSetFactory
from decision_rules.ruleset_factories._parsers import MLRulesParser
from decision_rules.ruleset_factories.utils.abstract_mlrules_factory import AbstractMLRulesRuleSetFactory


class MLRulesRuleSetFactory(AbstractMLRulesRuleSetFactory):
    """
    Factory for creating a ClassificationRuleSet from a list of lines of MLRules output file.

    Information about the MLRules algorithm and format can be found at:
    https://www.cs.put.poznan.pl/wkotlowski/software-mlrules.html

    Python wrapper:
    https://github.com/fracpete/mlrules-weka-package?tab=readme-ov-file

    Usage:
    see documentation of `make` method
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
            model: `MLRules`-type model as a list of lines from output file
            X_train: pandas dataframe with features
            y_train: data series with dependent variable
            measure_name: voting measure used to calculate rule voting weights

        Returns:
            ClassificationRuleSet: a set of classification
        """
        rules = MLRulesParser.parse(model)
        return TextRuleSetFactory().make(
            rules, X_train, y_train, measure_name=measure_name)
