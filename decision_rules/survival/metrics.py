"""Contains class for calculating rule metrics for survival rules
"""
from typing import Any
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd
from decision_rules.core.coverage import Coverage
from decision_rules.core.metrics import AbstractRulesMetrics
from decision_rules.survival.rule import SurvivalRule


class SurvivalRulesMetrics(AbstractRulesMetrics):
    """Class for calculating rule metrics for survival rules
    """

    @property
    def supported_metrics(self) -> list[str]:
        return list(self.get_metrics_calculator(None, None, None).keys())

    def get_metrics_calculator(
        self,
        rule: SurvivalRule,
        X: pd.DataFrame,
        y: pd.Series
    ) -> dict[str, Callable[[], Any]]:
        return {
            'p': lambda: int(rule.coverage.p),
            'n': lambda: int(rule.coverage.n),
            'P': lambda: int(rule.coverage.P),
            'N': lambda: int(rule.coverage.N),
            'unique': lambda: self._calculate_uniquely_covered_examples(
                rule, X, y, covered_type='all'
            ),
            "median_survival_time": lambda: float(rule.conclusion.value),
            "median_survival_time_ci_lower": lambda: float(rule.conclusion.median_survival_time_ci_lower),
            "median_survival_time_ci_upper": lambda: float(rule.conclusion.median_survival_time_ci_upper),
            "events_count": lambda: int(rule.conclusion.estimator.events_count_sum),
            "censored_count": lambda: int(rule.conclusion.estimator.censored_count_sum),
            "log_rank": lambda: float(rule.log_rank),
        }

    def calculate_p_value(self, coverage: Optional[Coverage] = None, rule: Optional[SurvivalRule] = None, y: Optional[np.ndarray] = None) -> float:
        raise NotImplementedError()
    
    def calculate_p_value(
        self,
        coverage: Optional[Coverage] = None,
        rule: Optional[SurvivalRule] = None,
        y: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculates the p-value for a survival rule based on the precomputed log_rank value.
            p_value = 1 - log_rank
        Args:
            coverage (Optional[Coverage]): Not used.
            rule (Optional[SurvivalRule]): The survival rule for which to calculate the p-value.
            y (Optional[np.ndarray]): Not used.

        Returns:
            float: The p-value for the rule.
        """
        if rule is None:
            raise ValueError("A survival rule must be provided to calculate p_value.")
        if rule.log_rank is None:
            raise ValueError(f"log_rank has not been computed for the rule with uuid: {rule.uuid}")
        return 1 - rule.log_rank