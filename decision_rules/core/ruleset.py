"""
Contains abstract ruleset class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

from decision_rules.conditions import AttributesCondition, CompoundCondition
from decision_rules.core.coverage import (ClassificationCoverageInfodict,
                                          Coverage)
from decision_rules.core.exceptions import InvalidStateError
from decision_rules.core.metrics import AbstractRulesMetrics
from decision_rules.core.prediction import PredictionStrategy, _PredictionModel
from decision_rules.core.rule import AbstractConclusion, AbstractRule
from decision_rules.measures import coverage, precision


class AbstractRuleSet(_PredictionModel, ABC):
    """Abstract ruleset allowing to perform prediction on data"""

    def __init__(self, rules: list[AbstractRule]) -> None:
        """
        Args:
            rules (list[AbstractRule]):
            dataset_metadata (BaseDatasetMetadata): metadata about datasets
                compatible with model
        """
        super().__init__()

        self.rules: list[AbstractRule] = rules
        self.column_names: list[str] = None
        self.train_P: dict[int, int] = None  # pylint: disable=invalid-name
        self.train_N: dict[int, int] = None  # pylint: disable=invalid-name
        self._default_conclusion: AbstractConclusion = None
        self._use_default_conclusion: bool = True
        self._stored_default_conclusion: AbstractConclusion = None
        self._prediction_strategy: Optional[PredictionStrategy] = None
        self.decision_attribute: Optional[str] = None

    @abstractmethod
    def get_metrics_object_instance(self) -> AbstractRulesMetrics:
        """Returns metrics object instance."""
        return None

    def _validate_object_state_before_prediction(self):
        voting_weights_calculated: bool = all(
            [rule.coverage is not None for rule in self.rules]
        )
        if not voting_weights_calculated:
            raise InvalidStateError(
                "Rule coverages have to be calculated before performing prediction."
                + "Did you forget to call update(...) method?"
            )

    @abstractmethod
    def _calculate_P_N(
        self, y_uniques: np.ndarray, y_values_count: np.ndarray
    ):  # pylint: disable=invalid-name
        pass

    def set_default_conclusion_enabled(self, enabled: bool) -> None:
        """Enable or disable usage of default conclusion during prediction.

        Args:
            enabled (bool): whether to use default conclusion or not
        """
        if self._stored_default_conclusion is None:
            raise InvalidStateError(
                "Default conclusion was never configured."
                + "Call update(...) method to calculate automatic default conclusion "
                + "or set it manually."
            )
        self._use_default_conclusion = enabled
        if enabled:
            self.default_conclusion = self._stored_default_conclusion
        else:
            self.default_conclusion = (
                self._stored_default_conclusion.__class__.make_empty(
                    self.decision_attribute
                )
            )

    @property
    def default_conclusion(self) -> AbstractConclusion:
        """Default conclusion used during prediction

        Returns:
            AbstractConclusion: default conclusion
        """
        return self._default_conclusion

    @default_conclusion.setter
    def default_conclusion(self, value: AbstractConclusion) -> None:
        self._default_conclusion = value
        if value is not None and not value.is_empty():
            self._use_default_conclusion = True
            self._stored_default_conclusion = value

    @property
    def is_using_default_conclusion(self) -> bool:
        """Whether default conclusion is enabled and used during prediction

        Returns:
            bool: whether default conclusion is enabled
        """
        return self._use_default_conclusion

    def _validate_coverage_matrix_param(
        self, X_binary: np.ndarray
    ):  # pylint: disable=invalid-name
        BASE_VALIDATION_ERROR_MESSAGE: str = (
            "Coverage matrix must be a 2D boolean numpy array."  # pylint: disable=invalid-name
        )
        if not isinstance(X_binary, np.ndarray):
            raise ValueError(
                BASE_VALIDATION_ERROR_MESSAGE
                + f' Passed object is of type "{str(type(X_binary))}".'
            )
        if len(X_binary.shape) != 2:
            raise ValueError(
                BASE_VALIDATION_ERROR_MESSAGE
                + f" Passed array have shape = {X_binary.shape}."
            )
        if X_binary.dtype != bool:
            raise ValueError(
                BASE_VALIDATION_ERROR_MESSAGE
                + f" Passed array have dtype {X_binary.dtype}."
            )

    def _sanitize_dataset(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        to_numpy: bool = True,
    ) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Sanitize and prepare dataset for other operations. Decision rules internally
        works on numpy arrays instead of pandas dataframes to improve execution times.
        This method transforms pandas dataframes into numpy arrays (if "to_numpy" is True) while
        preserving the columns order (self.column_names).

        Args:
            X (Union[np.ndarray, pd.DataFrame]): _description_
            y (Optional[Union[np.ndarray, pd.Series]], optional): _description_. Defaults to None.
            to_numpy (bool, optional): _description_. Defaults to True.

        Returns:
            Union[tuple[np.ndarray, np.ndarray], np.ndarray]: _description_
        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X[self.column_names] if self.column_names is not None else X
            if to_numpy:
                X = X.to_numpy()
        if isinstance(y, pd.Series) and to_numpy:
            y = y.to_numpy()
        return (X, y) if y is not None else X

    def calculate_coverage_matrix(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ):
        """
        Calculate binary coverage matrix showing if each rule covers each sample.
        Number of columns in the matrix is equal to the number of rules. Number
        of rows is equal to the number of samples in the dataset.
        Args:
            X (Union[np.ndarray, pd.DataFrame]): dataset
        """
        X: np.ndarray = self._sanitize_dataset(X)
        if len(self.rules) == 0:
            return np.empty(shape=(X.shape[0], 0), dtype=bool)
        coverage_matrix = np.array(
            [rule.premise.covered_mask(X) for rule in self.rules]
        ).T
        return coverage_matrix

    def calculate_rules_coverages(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> np.ndarray:
        """
        Args:
            X_train (Union[np.ndarray, pd.DataFrame]): train dataset
            y_train (Union[np.ndarray, pd.Series]): train labels

        Returns:
            np.ndarray: rules coverage matrix, same as calculated using "calculate_coverage_matrix"
            method.
        """
        X_train, y_train = self._sanitize_dataset(X_train, y_train)

        self._calculate_P_N(*np.unique(y_train, return_counts=True))

        coverage_matrix: np.ndarray = np.empty(
            shape=(X_train.shape[0], len(self.rules)), dtype=bool
        )
        for i, rule in enumerate(self.rules):
            P: int = (
                self.train_P[rule.conclusion.value]
                if self.train_P is not None
                else None
            )
            N: int = (
                self.train_N[rule.conclusion.value]
                if self.train_N is not None
                else None
            )
            with rule.premise.cache(recursive=False):
                rule.coverage = rule.calculate_coverage(
                    X_train, y_train, P=P, N=N, **kwargs
                )
                coverage_matrix[:, i] = rule.premise.covered_mask(X_train)
        return coverage_matrix

    def calculate_rules_weights(self, measure: Callable[[Coverage], float]):
        """
        Args:
            measure (Callable[[Coverage], float]): quality measure function

        Raises:
            ValueError: if any of the rules in ruleset has uncalculated coverage
        """
        for rule in self.rules:
            if rule.coverage is None:
                raise ValueError(
                    "Tried to calculate voting weight of a rule with uncalculated coverage."
                    + "You should either call `RuleSet.calculate_rules_coverages` method - to "
                    + "calculate coverages of all rules - or call `Rule.calculate_coverage` "
                    + "- to calculate coverage of this specific rule"
                )
            rule.voting_weight = measure(rule.coverage)
        self._voting_weights_calculated = True

    def _base_update(
        self,
        y_uniques: np.ndarray,
        y_values_count: np.ndarray,
        measure: Callable[[Coverage], float],
    ):
        self._calculate_P_N(y_uniques, y_values_count)
        self.calculate_rules_weights(measure)

        self._voting_weights_calculated = True

    def update_using_coverages(
        self,
        coverages_info: dict[str, ClassificationCoverageInfodict],
        measure: Callable[[Coverage], float],
        columns_names: list[str] = None,
    ):
        if self.column_names is None and columns_names is None:
            raise ValueError(
                'RuleSet has no configured "columns_names". Please pass "columns_names" list to this method.'
            )
        if len(self.rules) == 0:
            raise ValueError(
                '"update" cannot be called on empty ruleset with no rules.'
            )
        if len(self.rules) != len(coverages_info):
            raise ValueError(
                "Length of coverage_info should be the same as number "
                + f"of rules in ruleset ({len(self.rules)}), is: {len(coverages_info)}"
            )
        self.column_names = (
            columns_names if columns_names is not None else self.column_names
        )
        y_uniques: list[Any] = []
        y_values_count: list[Any] = []
        for rule in self.rules:
            try:
                coverage_info: ClassificationCoverageInfodict = coverages_info[
                    rule.uuid
                ]
                rule.coverage = Coverage(
                    p=coverage_info["p"],
                    n=coverage_info["n"],
                    P=coverage_info["P"],
                    N=coverage_info["N"],
                )
                if rule.conclusion.value not in y_uniques:
                    y_uniques.append(rule.conclusion.value)
                    y_values_count.append(rule.coverage.P)
            except KeyError:
                raise ValueError(  # pylint: disable=raise-missing-from
                    f'Coverage info missing for rule: "{rule.uuid}" '
                    + "and possibly some other rules too."
                )
        self._base_update(np.array(y_uniques), np.array(y_values_count), measure)

    def update(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        measure: Callable[[Coverage], float],
    ):
        """Updates ruleset using training dataset. This method should be called
        both after creation of new ruleset or after manipulating any of its rules
        or internal conditions. This method recalculates rules coverages and voting
        weights making it ready for prediction

        Args:
            X_train (pd.DataFrame):
            y_train (pd.Series):
            measure (Callable[[Coverage], float]): voting measure function

        Raises:
            ValueError: if called on empty ruleset with no rules
        """
        if len(self.rules) == 0:
            raise ValueError(
                '"update" cannot be called on empty ruleset with no rules.'
            )
        if self.column_names is None:
            self.column_names = X_train.columns.tolist()
        X_train, y_train = self._sanitize_dataset(X_train, y_train)
        y_uniques, y_values_count = np.unique(y_train, return_counts=True)
        coverage_matrix: np.ndarray = self.calculate_rules_coverages(X_train, y_train)

        self._base_update(y_uniques, y_values_count, measure)
        return coverage_matrix

    def predict(
        self,
        X: pd.DataFrame,  # pylint: disable=invalid-name
    ) -> np.ndarray:
        """
        Args:
            X (pd.DataFrame)
        Returns:
            np.ndarray: prediction
        """
        X: np.ndarray = self._sanitize_dataset(X)
        coverage_matrix: np.ndarray = self.calculate_coverage_matrix(X)
        self._validate_object_state_before_prediction()
        return self.predict_using_coverage_matrix(coverage_matrix)

    def calculate_rules_metrics(
        self,
        X: pd.DataFrame,  # pylint: disable=invalid-name
        y: pd.Series,  # pylint: disable=invalid-name
        metrics_to_calculate: Optional[list[str]] = None,
    ) -> dict[dict[str, str, float]]:
        """Calculate rules metrics for each rule such as precision,
        coverage, TP, FP etc. This method should be called after updating
        or calculating rules coverages.

        Args:
            X (pd.DataFrame):
            y (pd.Series):
            metrics_to_calculate (Optional[list[str]], optional): list of metrics names
                to calculate. Defaults to None.

        Raises:
            InvalidStateError: if rule's coverage have not been calculated
        Returns:
            dict[dict[str, str, float]]: metrics for each rule
        """
        old_coverages = [rule.coverage for rule in self.rules]
        old_conclusions = [rule.conclusion for rule in self.rules]
        for rule in self.rules:
            rule.premise.cached = True
        self.calculate_rules_coverages(X, y)

        metrics: AbstractRulesMetrics = self.get_metrics_object_instance()
        metrics_values: dict = metrics.calculate(X, y, metrics_to_calculate)

        for i, rule in enumerate(self.rules):
            rule.premise.invalidate_cache()
            rule.coverage = old_coverages[i]
            rule.conclusion = old_conclusions[i]

        return metrics_values

    @abstractmethod
    def calculate_condition_importances(
        self,
        X: pd.DataFrame,  # pylint: disable=invalid-name
        y: pd.Series,  # pylint: disable=invalid-name
        measure: Callable[[Coverage], float],
    ) -> Union[dict[str, float], dict[str, dict[str, float]]]:
        """Calculate importances of conditions in RuleSet

        Args:
            X (pd.DataFrame):
            y (pd.Series):
            measure (Callable[[Coverage], float]): measure used to count importance

        Returns:
            dict[str, float]: condition importances, in the case of classification additionally returns information about class dict[str, dict[str, float]]:
        """

    @abstractmethod
    def calculate_attribute_importances(
        self,
        condition_importances: Union[dict[str, float], dict[str, dict[str, float]]],
    ) -> Union[dict[str, float], dict[str, dict[str, float]]]:
        """Calculate importances of attriubtes in RuleSet based on conditions importances

        Args:
            condition_importances Union[dict[str, float], dict[str, dict[str, float]]]: condition
                importances

        Returns:
            dict[str, float]: attribute importances, in the case of classification additionally
                returns information about class dict[str, dict[str, float]]:
        """

    def calculate_ruleset_stats(self, *args, **kwargs) -> dict[str, float]:
        """Calculate ruleset statistics such as number of rules, average rule length,
        average precision, average coverage. This method should be called after updating
        rules coverages.

        Returns:
            dict: RuleSet statistics
        """

        def count_conditions(condition) -> int:
            """Recursively count the leaf subconditions."""
            if not condition.subconditions:
                return 1
            return sum(
                count_conditions(subcondition)
                for subcondition in condition.subconditions
            )

        stats = dict()
        stats["rules_count"] = len(self.rules)
        stats["avg_conditions_count"] = round(
            np.mean([len(rule.premise.subconditions) for rule in self.rules]), 2
        )
        stats["avg_precision"] = round(
            np.mean([precision(rule.coverage) for rule in self.rules]), 2
        )
        stats["avg_coverage"] = round(
            np.mean([coverage(rule.coverage) for rule in self.rules]), 2
        )
        stats["total_conditions_count"] = sum(
            [count_conditions(rule.premise) for rule in self.rules]
        )

        return stats

    def local_explainability(
        self, x: pd.Series
    ) -> tuple[list[str], str]:  # pylint: disable=invalid-name
        """Calculate local explainability of ruleset for given instance.

        Args:
            x (pd.Series): Instance to explain

        Returns:
            list: list of rules covering instance
            str: Decision (in classification task) or prediction (in regression task)
        """
        x: np.ndarray = self._sanitize_dataset(x)
        x = x.reshape(1, -1)
        # TODO: here we can cache rules covered masks for improved performance
        prediction: np.ndarray = self.predict(x)
        rules_covering_instance = [
            rule.uuid
            for rule in self.rules
            if np.sum(rule.premise.covered_mask(x)) == 1
        ]
        return rules_covering_instance, prediction

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AbstractRuleSet) and other.rules == self.rules

    def split_dataset(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = dataset.drop(columns=self.decision_attribute)
        y = dataset[self.decision_attribute]
        return X, y

    @property
    def coverage_dict(self) -> dict:
        return {rule.uuid: rule.get_coverage_dict() for rule in self.rules}

    def update_meta(self, new_attributes: list[str]):
        if set(self.column_names).difference(set(new_attributes)):
            raise ValueError("New attributes do not contain all of the old attributes.")
        old_to_new_attr_mapping = {
            i: new_attributes.index(attr) for i, attr in enumerate(self.column_names)
        }
        for rule in self.rules:
            self._update_condition(rule.premise, old_to_new_attr_mapping)
            rule.column_names = new_attributes
        self.column_names = new_attributes

    def _update_condition(self, premise, old_to_new_attr_mapping):
        if isinstance(premise, CompoundCondition):
            for condition in premise.subconditions:
                self._update_condition(condition, old_to_new_attr_mapping)
            return
        if isinstance(premise, AttributesCondition):
            premise.column_left = old_to_new_attr_mapping[premise.column_left]
            premise.column_right = old_to_new_attr_mapping[premise.column_right]
            return
        premise.column_index = old_to_new_attr_mapping[premise.column_index]
