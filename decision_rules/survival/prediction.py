from __future__ import annotations

from typing import Any
from typing import Optional
from typing import TypedDict

import numpy as np
from decision_rules.core.prediction import \
    BestRulePredictionStrategy as BaseBestRulePredictionStrategy
from decision_rules.core.prediction import PredictionStrategy
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator
from decision_rules.survival.kaplan_meier import SurvInfo
from decision_rules.survival.rule import SurvivalConclusion


class SurvivalPrediction(TypedDict):
    """Object describing survival prediction. It contains
    times and probabilities of the predicted Kaplan-Meier curve.
    It also contains median survival time.
    """
    times: np.ndarray
    probabilities: np.ndarray
    median_survival_time: float

    @staticmethod
    def from_kaplan_meier(km: Optional[KaplanMeierEstimator]) -> Optional[SurvivalPrediction]:
        if km is None:
            return None
        return {
            "times": km.times,
            "probabilities": km.probabilities,
            "median_survival_time": km.median_survival_time
        }

    def to_kaplan_meier(self) -> Optional[KaplanMeierEstimator]:
        if self is None:
            return None
        return KaplanMeierEstimator(SurvInfo(
            time=self['times'],
            probability=self['probabilities'],
            events_count=np.zeros(self['times'].shape),
            censored_count=np.zeros(self['times'].shape),
            at_risk_count=np.zeros(self['times'].shape),
        ))


class VotingPredictionStrategy(PredictionStrategy):
    """Voting prediction strategy for survival prediction.

    Based on article: Wróbel et al. Learning rule sets from survival data BMC Bioinformatics (2017) 18:285 Page 5 of 13
    The learned rule set can be applied for an estimation of the survival function of new observations based on the values taken by their covariates.
    The estimation is performed by rules covering given observation. If observation is not covered by any of the rules then it has assigned the default survival estimate computed on the entire train ing set.
    Otherwise, final survival estimate is calculated as an average of survival estimates of all rules covering the observation
    """

    def _perform_prediction(self, voting_matrix: np.ndarray) -> np.ndarray:
        # Based on article: Wróbel et al. Learning rule sets from survival data BMC Bioinformatics (2017) 18:285 Page 5 of 13
        # The learned rule set can be applied for an estimation of the survival function of new observations based on the values taken by their covariates.
        # The estimation is per formed by rules covering given observation. If observation is not covered by any of the rules then it has assigned the default survival estimate computed on the entire train ing set.
        # Otherwise, final survival estimate is calculated as an average of survival estimates of all rules covering the observation
        num_examples = voting_matrix.shape[0]
        # create numpy array with rules for easier indexing and masking
        conclusions_array: np.ndarray = np.array([
            r.conclusion.estimator for r in self.rules
        ])
        prediction_array: np.ndarray = np.empty(
            (num_examples,), dtype=object
        )
        for i in range(num_examples):
            km: KaplanMeierEstimator = self._predict_for_example(
                example_index=i,
                coverage_matrix=self.coverage_matrix,
                conclusions_array=conclusions_array
            )
            prediction_array[i] = km
        return prediction_array

    def _predict_for_example(
        self,
        example_index: int,
        coverage_matrix: np.ndarray,
        conclusions_array: np.ndarray
    ) -> KaplanMeierEstimator:
        """Return a predicted  Kaplan-Meier estimator for a single example.

        Args:
            example_index (int): sample index
            coverage_mask (np.ndarray, optional): rules coverage mask

        Returns:
            KaplanMeierEstimator: _description_
        """
        matching_rules_conclusions: np.ndarray = conclusions_array[
            coverage_matrix[example_index, :]
        ]
        if len(matching_rules_conclusions) == 0:
            estimator: KaplanMeierEstimator = self.default_conclusion.estimator
        else:
            estimator: KaplanMeierEstimator = KaplanMeierEstimator.average(
                matching_rules_conclusions
            )
        return estimator


class BestRulePredictionStrategy(BaseBestRulePredictionStrategy):
    """Best rule prediction strategy for survival prediction.
    """

    def _get_prediction_from_conclusion(self, conclusion: SurvivalConclusion) -> Any:
        return conclusion.estimator
