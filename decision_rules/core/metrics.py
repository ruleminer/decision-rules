"""
Contains base abstract class for calculating rules metrics.
"""
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd
from decision_rules.conditions import CompoundCondition
from decision_rules.core.condition import AbstractCondition
from decision_rules.core.coverage import Coverage
from decision_rules.core.rule import AbstractRule


class AbstractRulesMetrics(ABC):
    """Abstract class for rules metrics calculations. All classes calculating
    metrics for different types of rulesets should inherit this class.

    Args:
        ABC (_type_):
    """

    def __init__(self, rules: list[AbstractRule]) -> None:
        super().__init__()
        self.rules: list[AbstractRule] = rules

    @abstractmethod
    def get_metrics_calculator(
        self,
        rule: AbstractRule,
        X: pd.DataFrame,
        y: pd.Series
    ) -> dict[str, Callable[[], Any]]:
        """Returns metrics calculator object in a form of dictionary where
        values are the non-parmetrized callables calculating specified metrics
        and keys are the names of those metrics.

        Examples
        --------
        >>> {
        >>>  'p': lambda: rule.coverage.p,
        >>>  'n': lambda: rule.coverage.n,
        >>>  'P': lambda: rule.coverage.P,
        >>>  'N': lambda: rule.coverage.N,
        >>>  'coverage': lambda: measures.coverage(rule.coverage),
        >>>  ...
        >>> }


        Args:
            rule (AbstractRule): rule
            X (pd.DataFrame): data
            y (pd.Series): labels

        Returns:
            dict[str, Callable[[], Any]]: metrics calculator object
        """
    @abstractmethod
    def calculate_p_value(self, coverage: Optional[Coverage] = None, rule: Optional[AbstractRule] = None, y: Optional[np.ndarray] = None) -> float:
        """Abstract method to calculate p-value

        Args:
            coverage (Optional[Coverage], optional): Coverage object for classification rules. Defaults to None.
            rule (Optional[RegressionRule], optional): The rule from regression ruleset for which p-value is to be calculated.. Defaults to None.
            y (Optional[np.ndarray], optional): Target labels for regression rules. Defaults to None.
        """
        pass

    @property
    @abstractmethod
    def supported_metrics(self) -> list[str]:
        """
        Returns:
            list[str]: list of names of all supported metrics
        """

    def _calculate_uniquely_covered_examples_in_pos_and_neg(
        self,
        rule: AbstractRule,
        X: pd.DataFrame,  # pylint: disable=invalid-name
        y: pd.Series,  # pylint: disable=invalid-name
        covered_type: str
    ) -> int:
        """Calculates uniquely covered examples for a given rule. Either positive
            or negative based on covered_type param.

        Args:
            rule (AbstractRule): rule
            X (pd.DataFrame):
            y (pd.Series):
            covered_type (str): Parameter specifying whether it should calculate
                positive covered examples or negative covered examples.
                Parameter values should be either 'positive' or 'negative'.

        Raises:
            ValueError: when covered_type is not 'positive' or 'negative'

        Returns:
            int: Number of uniquely covered examples
        """
        other_rules: list[AbstractRule] = [
            other_rule for other_rule in self.rules if other_rule.uuid != rule.uuid
        ]
        if covered_type == 'positive':
            rules_covered_masks: dict[str, np.ndarray] = {
                other_rule.uuid: other_rule.positive_covered_mask(X, y)
                for other_rule in other_rules
            }
        elif covered_type == 'negative':
            rules_covered_masks: dict[str, np.ndarray] = {
                other_rule.uuid: other_rule.negative_covered_mask(X, y)
                for other_rule in other_rules
            }
        else:
            raise ValueError(
                '"covered_type" parameter should be either "positive" or "negative"')

        if rules_covered_masks:
            others_rules_covered_mask = np.logical_or.reduce(
                list(rules_covered_masks.values()))
        else:
            others_rules_covered_mask = np.zeros(shape=y.shape, dtype=bool)

        if covered_type == 'positive':
            current_rule_mask = rule.positive_covered_mask(X, y)
        else:
            current_rule_mask = rule.negative_covered_mask(X, y)

        unique_mask = current_rule_mask & ~others_rules_covered_mask

        return int(np.count_nonzero(unique_mask))

    def _calculate_uniquely_covered_examples(
        self,
        rule: AbstractRule,
        X: pd.DataFrame,
        y: pd.Series,
        covered_type: str
    ) -> int:
        """Calculates the number of uniquely covered examples for a given rule,
        where uniqueness means that no other rule covers the example,
        regardless of the predicted class.

        Args:
            rule (AbstractRule): The rule for which to calculate uniquely covered examples.
            X (pd.DataFrame): The dataset features.
            y (pd.Series): The dataset labels.
            covered_type (str): 'positive' or 'negative'

        Returns:
            int: Number of uniquely covered examples of the specified type.
        """
        other_rules: list[AbstractRule] = [
            other_rule for other_rule in self.rules if other_rule.uuid != rule.uuid
        ]

        # Other rules' total coverage masks
        rules_covered_masks = [
            other_rule.premise.covered_mask(X)
            for other_rule in other_rules
        ]

        if rules_covered_masks:
            others_rules_covered_mask = np.logical_or.reduce(
                rules_covered_masks)
        else:
            others_rules_covered_mask = np.zeros(shape=y.shape, dtype=bool)

        # Current rule's positive or negative or all covered mask
        if covered_type == 'positive':
            current_rule_mask = rule.positive_covered_mask(X, y)
        elif covered_type == 'negative':
            current_rule_mask = rule.negative_covered_mask(X, y)
        elif covered_type == 'all':
            current_rule_mask = rule.premise.covered_mask(X)
        else:
            raise ValueError(
                '"covered_type" parameter should be either "positive" or "negative"')

        # Uniquely covered examples
        unique_mask = current_rule_mask & ~others_rules_covered_mask

        return int(np.count_nonzero(unique_mask))

    def _calculate_conditions_count(self, rule: AbstractRule) -> int:
        def calculate_conditions_count_recursive(condition: AbstractCondition) -> int:
            if isinstance(condition, CompoundCondition):
                return sum([
                    calculate_conditions_count_recursive(subcondition)
                    for subcondition in condition.subconditions
                ])
            return 1

        return calculate_conditions_count_recursive(rule.premise)

    def calculate(
        self,
        X: pd.DataFrame,  # pylint: disable=invalid-name
        y: pd.Series,  # pylint: disable=invalid-name
        metrics_to_calculate: Optional[list[str]] = None
    ) -> dict[str, dict[str, float]]:
        """Calculates rules metrics for all rules

        Args:
            X (pd.DataFrame):
            y (pd.Series):
            metrics_to_calculate (Optional[list[str]], optional): Optional parameter
                for specifying which metrics to calculate. By default it will calculate
                all supported metrics.

        Raises:
            ValueError: when trying to calculate unsupported metric.

        Returns:
            dict[str, dict[str, float]]: Dictionary of metrics where keys are rules uuids
                and values are dictionaries containing metrics values for this rules.
        """
        if metrics_to_calculate is None:
            metrics_to_calculate = list(
                self.get_metrics_calculator(None, X, y).keys())
        metrics: dict[str, dict[str, Any]] = {
            rule.uuid: {} for rule in self.rules
        }
        try:
            for rule in self.rules:
                calculator: dict[Callable[[], Any]] = self.get_metrics_calculator(
                    rule, X, y
                )
                metrics[rule.uuid] = {
                    metric_name: calculator[metric_name]()
                    for metric_name in metrics_to_calculate
                }
        except KeyError as error:
            raise ValueError(
                f'Unsupported metrics: "{metrics_to_calculate}". ' +
                'Supported metrics for this type of ruleset are: ' +
                f'{", ".join(self.supported_metrics)}'
            ) from error
        return metrics
