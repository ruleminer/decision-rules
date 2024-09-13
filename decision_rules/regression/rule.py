"""
Contains regression rule and conclusion classes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from decision_rules import settings
from decision_rules.core.condition import AbstractCondition
from decision_rules.core.coverage import Coverage
from decision_rules.core.rule import AbstractConclusion
from decision_rules.core.rule import AbstractRule


class RegressionConclusion(AbstractConclusion):
    """Conclusion part of the regression rule

    Args:
        AbstractConclusion (_type_):
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        value: float,
        column_name: str,
        fixed: bool = False,
        low: Optional[float] = None,
        high: Optional[float] = None,
    ) -> None:
        self._validate(value, low, high)
        self.fixed: bool = fixed
        self._value: float = value
        self.low: float = low
        self.high: float = high
        self.column_name: str = column_name\

        self.train_covered_y_mean: float = None
        self.train_covered_y_std: float = None
        self.train_covered_y_min: float = None
        self.train_covered_y_max: float = None

    def _validate(self, value: float, low: float, high: float):
        if low is not None and low > value:
            raise ValueError(
                'Low boundary cannot be greater than conclusion value, got: ' +
                f'value = {value} and low = {low}'
            )
        if high is not None and high < value:
            raise ValueError(
                'High boundary cannot be lower than conclusion value, got: ' +
                f'value = {value} and high = {high}'
            )

    @property
    def value(self) -> float:
        """
        Returns:
            float: Conclusion's value
        """
        return self._value

    @value.setter
    def value(self, value: float):
        self._value = value

    def calculate_low_high(
        self,
    ):
        """Automatically calculate low and high values based on
        standard deviation of covered examples label attribute
        where:
            low = value - train_covered_y_std
            high = value + train_covered_y_std

        Args:
            train_covered_y_std (float): standard deviation of covered examples label attribute

        Raises:
            ValueError: When conclusion value is None
        """
        if self._value is not None and self.train_covered_y_std is not None:
            self.low = self._value - self.train_covered_y_std
            self.high = self._value + self.train_covered_y_std
        else:
            raise ValueError(
                'Cannot calculate low and high values if value is None or train_covered_y_std is None'
            )

    def positives_mask(self, y: np.ndarray) -> np.ndarray:
        return ((y >= self.low) & (y <= self.high))

    @staticmethod
    def make_empty(column_name: str) -> RegressionConclusion:  # pylint: disable=invalid-name
        return RegressionConclusion(
            column_name=column_name,
            value=np.nan,
            fixed=True
        )

    def is_empty(self) -> bool:
        return np.isnan(self.value)

    def __hash__(self) -> int:
        return hash((self._value, self.low, self.high, self.column_name))

    def __str__(self) -> str:
        return (
            f'{self.column_name} = {{{self._value:,.{settings.FLOAT_DISPLAY_PRECISION}}}} ' +
            f'[{self.low:,.{settings.FLOAT_DISPLAY_PRECISION}}, ' +
            f'{self.high:,.{settings.FLOAT_DISPLAY_PRECISION}}]'
        )


class RegressionRule(AbstractRule):
    """Regression rule.
    """

    def __init__(
        self,
        premise: AbstractCondition,
        conclusion: RegressionConclusion,
        column_names: list[str]
    ) -> None:
        self.conclusion: RegressionConclusion = conclusion
        super().__init__(premise, conclusion, column_names)

        self.train_covered_y_mean: float = None

    def calculate_coverage(
            self,
            X: np.ndarray,
            y: np.ndarray = None,
            P: int = None,
            N: int = None
    ) -> Coverage:
        covered_y: np.ndarray = y[self.premise.covered_mask(X)]
        self.conclusion.train_covered_y_mean = np.mean(covered_y)

        y_mean: float = np.mean(covered_y)
        self.conclusion.train_covered_y_std: float = np.sqrt(
            (np.sum(np.square(covered_y)) /
             covered_y.shape[0]) - (y_mean * y_mean)
        )
        self.conclusion.train_covered_y_mean: float = np.mean(covered_y)
        # np.min and max will fail badly on empty arrays
        if covered_y.shape[0] == 0:
            self.conclusion.train_covered_y_min: float = np.nan
            self.conclusion.train_covered_y_max: float = np.nan
        else:
            self.conclusion.train_covered_y_min: float = np.min(covered_y)
            self.conclusion.train_covered_y_max: float = np.max(covered_y)

        if not self.conclusion.fixed:
            self.conclusion.value = self.conclusion.train_covered_y_mean
            self.conclusion.calculate_low_high()
        return super().calculate_coverage(X, y, P, N)

    def get_coverage_dict(self) -> dict:
        coverage = super().get_coverage_dict()
        coverage["train_covered_y_std"] = self.conclusion.train_covered_y_std
        coverage["train_covered_y_mean"] = self.conclusion.train_covered_y_mean
        coverage["train_covered_y_min"] = self.conclusion.train_covered_y_min
        coverage["train_covered_y_max"] = self.conclusion.train_covered_y_max
        return coverage
