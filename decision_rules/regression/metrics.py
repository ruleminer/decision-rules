"""Contains class for calculating rule metrics for regression rules
"""
import math
from typing import Any
from typing import Callable
from typing import Optional

import numpy as np
from decision_rules.core.coverage import Coverage
from decision_rules.core.metrics import AbstractRulesMetrics
from decision_rules.regression.rule import RegressionRule
from scipy.stats import chi2
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error


class RegressionRulesMetrics(AbstractRulesMetrics):
    """
    Returns metrics object instance."""

    def __init__(self, rules: list[RegressionRule]):
        super().__init__(rules)

        self.expected_dev: float = None

    @property
    def supported_metrics(self) -> list[str]:
        return list(self.get_metrics_calculator(None, None, None).keys())

    def get_metrics_calculator(
        self,
        rule: RegressionRule,
        X: np.ndarray,
        y: np.ndarray
    ) -> dict[str, Callable[[], Any]]:
        rule_covered_examples: np.ndarray = np.array([])
        rule_prediction: np.ndarray = np.array([])
        if rule is not None:
            rule_prediction = np.full(
                shape=rule.premise.covered_mask(X).shape,
                fill_value=rule.conclusion.value
            )
            rule_covered_examples = y[rule.premise.covered_mask(X)]
        return {
            'p': lambda: int(rule.coverage.p),
            'n': lambda: int(rule.coverage.n),
            'P': lambda: int(rule.coverage.P),
            'N': lambda: int(rule.coverage.N),
            'p_unique': lambda: self._calculate_uniquely_covered_examples(
                rule, X, y, covered_type='positive'
            ),
            'n_unique': lambda: self._calculate_uniquely_covered_examples(
                rule, X, y, covered_type='negative'
            ),
            'support': lambda: int(rule.coverage.p + rule.coverage.n),
            'conditions_count': lambda: int(self._calculate_conditions_count(rule)),
            'y_covered_avg': lambda: float(rule_covered_examples.mean()),
            'y_covered_median': lambda: float(np.median(rule_covered_examples)),
            'y_covered_min': lambda: float(rule_covered_examples.min()),
            'y_covered_max': lambda: float(rule_covered_examples.max()),
            'mae': lambda: float(mean_absolute_error(y, rule_prediction)),
            'rmse': lambda: float(math.sqrt(mean_squared_error(y, rule_prediction))),
            'mape': lambda: float(mean_absolute_percentage_error(y, rule_prediction)),
            'p-value': lambda: float(self.calculate_p_value(rule=rule, y=y)),
        }

    def calculate_p_value(self, coverage: Optional[Coverage] = None, rule: Optional[RegressionRule] = None, y: Optional[np.ndarray] = None) -> float:
        """Calculates ryle p-value on given dataset based on X2 test
        comparing label variance of covered vs. uncovered examples.

        Args:
            rule (RegressionRule): rule
            y (np.ndarray): labels

        Returns:
            float: rule p-value
        """
        expected_dev = math.sqrt(y.var())
        sample_size: int = rule.coverage.p + rule.coverage.n
        factor: float = rule.conclusion.train_covered_y_std / expected_dev
        t: float = (  # pylint: disable=invalid-name
            float(sample_size - 1) *
            (factor * factor)
        )
        p_value: float = chi2.cdf(t, df=sample_size - 1)
        return p_value
