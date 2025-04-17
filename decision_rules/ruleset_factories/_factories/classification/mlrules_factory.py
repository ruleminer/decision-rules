from typing import Callable
from typing import Union
from typing import Any
from typing import Iterable

import numpy as np
import pandas as pd
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.ruleset_factories._factories.classification.text_factory import TextRuleSetFactory
from decision_rules.ruleset_factories._parsers import MLRulesParser
from decision_rules.ruleset_factories.utils.abstract_mlrules_factory import AbstractMLRulesRuleSetFactory
from decision_rules.core.exceptions import InvalidMeasureNameException
from decision_rules.core.exceptions import MLRulesParsingException
from decision_rules.helpers import get_measure_function_by_name


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
        labels_values, y_counts = np.unique(y_train, return_counts=True)

        ruleset = self._build_ruleset(
            model,
            y_counts,
            decision_attribute_name=y_train.name,
            labels_values=labels_values,
            columns_names=X_train.columns.tolist()
        )

        if isinstance(measure_name, str):
            measure = get_measure_function_by_name(measure_name)
        elif callable(measure_name):
            measure = measure_name
        else:
            raise InvalidMeasureNameException()
        ruleset.y_values = labels_values
        ruleset.update(
            X_train, y_train,
            measure=measure
        )


        return ruleset

    def _build_ruleset(
        self,
        model: list[str],
        y_counts: np.ndarray,
        decision_attribute_name: str,
        labels_values: Iterable[Any],
        columns_names: list[str]
    ) -> ClassificationRuleSet:

        try:
            rules = MLRulesParser.parse(model)
        except Exception as e:
            raise MLRulesParsingException(e) from None

        ruleset = TextRuleSetFactory()._build_ruleset(
            rules,
            y_counts,
            decision_attribute_name=decision_attribute_name,
            labels_values=labels_values,
            columns_names=columns_names
        )

        return ruleset
