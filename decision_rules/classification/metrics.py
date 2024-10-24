"""Contains class for calculating rule metrics for classification rules
"""
from typing import Any
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd
from decision_rules import measures
from decision_rules.classification.rule import ClassificationRule
from decision_rules.core.coverage import Coverage
from decision_rules.core.metrics import AbstractRulesMetrics
from scipy.stats import fisher_exact
from scipy.stats import hypergeom


class ClassificationRulesMetrics(AbstractRulesMetrics):
    """Class for calculating rule metrics for classification rules
    """

    @property
    def supported_metrics(self) -> list[str]:
        return list(self.get_metrics_calculator(None, None, None).keys())

    def get_metrics_calculator(
        self,
        rule: ClassificationRule,
        X: pd.DataFrame,
        y: pd.Series
    ) -> dict[str, Callable[[], Any]]:
        return {
            'p': lambda: int(rule.coverage.p),
            'n': lambda: int(rule.coverage.n),
            'P': lambda: int(rule.coverage.P),
            'N': lambda: int(rule.coverage.N),
            'unique_in_pos': lambda: self._calculate_uniquely_covered_examples_in_pos_and_neg(
                rule, X, y, covered_type='positive'
            ),
            'unique_in_neg': lambda: self._calculate_uniquely_covered_examples_in_pos_and_neg(
                rule, X, y, covered_type='negative'
            ),
            'p_unique': lambda: self._calculate_uniquely_covered_examples(
                rule, X, y, covered_type='positive'
            ),
            'n_unique': lambda: self._calculate_uniquely_covered_examples(
                rule, X, y, covered_type='negative'
            ),
            'all_unique': lambda: self._calculate_uniquely_covered_examples(
                rule, X, y, covered_type='all'
            ),
            'support': lambda: float((rule.coverage.p + rule.coverage.n) / (rule.coverage.P + rule.coverage.N)),
            'conditions_count': lambda: int(self._calculate_conditions_count(rule)),
            'precision': lambda: float(measures.precision(rule.coverage)),
            'coverage': lambda: float(measures.coverage(rule.coverage)),
            'C2': lambda: float(measures.c2(rule.coverage)),
            'RSS': lambda: float(measures.rss(rule.coverage)),
            'correlation': lambda: float(measures.correlation(rule.coverage)),
            'lift': lambda: float(measures.lift(rule.coverage)),
            'p_value': lambda: float(self.calculate_p_value(coverage=rule.coverage)),
            'TP': lambda: int(rule.coverage.p),
            'FP': lambda: int(rule.coverage.n),
            'TN': lambda: int(rule.coverage.N - rule.coverage.n),
            'FN': lambda: int(rule.coverage.P - rule.coverage.p),
            'sensitivity': lambda: float(measures.sensitivity(rule.coverage)),
            'specificity': lambda: float(measures.specificity(rule.coverage)),
            'negative_predictive_value': lambda: float(self._calculate_negative_predictive_value(rule)),
            'odds_ratio': lambda: float(measures.odds_ratio(rule.coverage)),
            'relative_risk': lambda: float(measures.relative_risk(rule.coverage)),
            'lr+': lambda: self._calculate_lr_plus(rule),
            'lr-': lambda: self._calculate_lr_minus(rule),
        }

    def _calculate_lr_plus(self, rule: ClassificationRule) -> float:
        """Calculates likelihood ratio positive

        Args:
            rule (ClassificationRule): rule

        Returns:
            float: likelihood ratio positive
        """
        denominator = 1 - measures.specificity(rule.coverage)
        if denominator == 0.0:
            return float('inf')
        return float(measures.sensitivity(rule.coverage) / denominator)

    def _calculate_lr_minus(self, rule: ClassificationRule) -> float:
        """Calculates likelihood ratio negative

        Args:
            rule (ClassificationRule): rule

        Returns:
            float: likelihood ratio negative
        """
        denominator = measures.specificity(rule.coverage)
        if denominator == 0.0:
            return float('inf')
        return float((1 - measures.sensitivity(rule.coverage)) / denominator)

    def _calculate_negative_predictive_value(self, rule: ClassificationRule) -> float:
        """Calculates relative number of correctly as negative classified
        examples among all examples classified as negative

        Args:
            rule (ClassificationRule): rule

        Returns:
            float: negative_predictive_value
        """
        coverage: Coverage = rule.coverage
        tn: int = coverage.N - coverage.n
        fn: int = coverage.P - coverage.p
        return tn / (fn + tn)

    def calculate_p_value(self, coverage: Optional[Coverage] = None, rule: Optional[ClassificationRule] = None, y: Optional[np.ndarray] = None) -> float:
        """Calculates Fisher's exact test for confusion matrix

        Args:
            coverage (Coverage): coverage

        Returns:
            float: p_value
        """
        confusion_matrix = np.array([
            # TP, FP
            [coverage.p, coverage.n],
            # FN, TN
            [coverage.P - coverage.p, coverage.N - coverage.n]]
        )
        _, p_value = fisher_exact(confusion_matrix)
        return p_value
