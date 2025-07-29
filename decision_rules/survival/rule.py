"""
Contains survival rule and conclusion classes.
"""
from __future__ import annotations

import numpy as np
from decision_rules import settings
from decision_rules.core.condition import AbstractCondition
from decision_rules.core.coverage import Coverage
from decision_rules.core.rule import AbstractConclusion
from decision_rules.core.rule import AbstractRule
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator


class SurvivalConclusion(AbstractConclusion):
    """Conclusion part of the survival rule

    Args:
        AbstractConclusion (_type_):
    """

    def __init__(
        self,
        value: float,  # median_survival_time
        column_name: str,
        fixed: bool = False
    ) -> None:
        super().__init__(value, column_name)
        self.fixed: bool = fixed
        self._estimator = KaplanMeierEstimator()
        self.median_survival_time_ci_lower = None
        self.median_survival_time_ci_upper = None

    @property
    def estimator(self) -> KaplanMeierEstimator:
        """
        Returns:
            KaplanMeierEstimator: KaplanMeierEstimator
        """
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: KaplanMeierEstimator):
        self._estimator = estimator

    def positives_mask(self, y: np.ndarray) -> np.ndarray:
        # Based on article: Wróbel et al. Learning rule sets from survival data BMC Bioinformatics (2017) 18:285 Page 4 of 13
        # An observation is covered by the rule when it satisfies its premise. The conclusion of r is an estimate Sˆ(T|cj) of the survival function.
        # Particularly, it is a Kaplan-Meier (KM) estimator [50] calculated on the basis of the instances covered by the rule, that is, satisfying all conditions cj (j = 1, ... , n)

        # Macha: So i think that conclusion positive mask should alwasy be true
        return np.ones(y.shape[0], dtype=bool)

    @staticmethod
    def make_empty(column_name: str) -> SurvivalConclusion:  # pylint: disable=invalid-name
        conclusion = SurvivalConclusion(
            column_name=column_name,
            value=None,
            fixed=True
        )
        conclusion.estimator = None
        return conclusion

    def is_empty(self) -> bool:
        return self.value is None

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AbstractConclusion) and
            other.column_name == self.column_name and
            other.value == self.value
        )

    def __hash__(self) -> int:
        return hash((self.column_name, self.value))

    def __str__(self) -> str:
        return f'{self.column_name} = {{{self.value}}}'


class SurvivalRule(AbstractRule):
    """Survival decision rule.
    """

    def __init__(
        self,
        premise: AbstractCondition,
        conclusion: SurvivalConclusion,
        column_names: list[str],
        survival_time_attr: str = None
    ) -> None:
        self.conclusion: SurvivalConclusion = conclusion
        self.measure = None
        super().__init__(premise, conclusion, column_names)
        if survival_time_attr is not None:
            self.survival_time_attr = survival_time_attr
            self.survival_time_attr_idx = self.column_names.index(
                survival_time_attr)
        else:
            self.survival_time_attr = None
            self.survival_time_attr_idx = None

    def set_survival_time_attr(self, survival_time_attr: str):
        self.survival_time_attr = survival_time_attr
        self.survival_time_attr_idx = self.column_names.index(
            survival_time_attr)

    def calculate_coverage(
            self,
            X: np.ndarray,
            y: np.ndarray = None,
            P: int = None,
            N: int = None,
            **kwargs
    ) -> Coverage:
        if self.survival_time_attr_idx is None:
            raise ValueError(
                'Survival time attribute is not set for this rule. Please call set_survival_time_attr method before calculating coverage')

        with self.premise.cache():
            covered_mask = self.premise.covered_mask(X)
            uncovered_mask = self.premise.uncovered_mask(X)

        if not self.conclusion.fixed:
            self.conclusion.estimator.fit(
                X[covered_mask, self.survival_time_attr_idx],
                y[covered_mask],
                skip_sorting=kwargs.get('skip_sorting', False)
            )
        covered_examples_indexes = np.where(covered_mask)[0]
        uncovered_examples_indexes = np.where(uncovered_mask)[0]
        survival_time: np.ndarray = X[:, self.survival_time_attr_idx]
        self.log_rank, self.log_rank_stats = KaplanMeierEstimator.log_rank(
            survival_time,
            y,
            covered_examples_indexes,
            uncovered_examples_indexes,
            return_stats=True,
        )
        self.conclusion.value = self.conclusion.estimator.median_survival_time
        self.conclusion.median_survival_time_ci_lower = self.conclusion.estimator.median_survival_time_cli.iloc[
            0]["prob_lower_0.95"]
        self.conclusion.median_survival_time_ci_upper = self.conclusion.estimator.median_survival_time_cli.iloc[
            0]["prob_upper_0.95"]
        return super().calculate_coverage(X, y, P, N)

    def get_coverage_dict(self) -> dict:
        coverage = super().get_coverage_dict()
        coverage["median_survival_time"] = self.conclusion.value
        coverage["median_survival_time_ci_lower"] = self.conclusion.median_survival_time_ci_lower
        coverage["median_survival_time_ci_upper"] = self.conclusion.median_survival_time_ci_upper
        coverage["events_count"] = self.conclusion.estimator.events_count_sum
        coverage["censored_count"] = self.conclusion.estimator.censored_count_sum
        coverage["log_rank"] = self.log_rank
        coverage["kaplan_meier_estimator"] = {}
        coverage["kaplan_meier_estimator"]["times"] = self.conclusion.estimator.times.tolist()
        coverage["kaplan_meier_estimator"]["events_count"] = self.conclusion.estimator.events_counts.tolist()
        coverage["kaplan_meier_estimator"]["censored_count"] = self.conclusion.estimator.censored_counts.tolist()
        coverage["kaplan_meier_estimator"]["at_risk_count"] = self.conclusion.estimator.at_risk_counts.tolist()
        coverage["kaplan_meier_estimator"]["probabilities"] = self.conclusion.estimator.probabilities.tolist()
        return coverage
