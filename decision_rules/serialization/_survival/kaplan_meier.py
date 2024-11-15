"""
Contains classes for Kaplan-Meier JSON serialization.
"""
from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel

from decision_rules.survival.kaplan_meier import (KaplanMeierEstimator,
                                                  KaplanMeierEstimatorDict)
from decision_rules.survival.rule import SurvivalConclusion


class _KaplanMeierEstimatorModel(BaseModel):
    times: list[Union[int, float]]
    events_count: list[int]
    censored_count: list[int]
    at_risk_count: list[int]
    probabilities: list[float]

    value: Optional[float]

    @staticmethod
    def from_rule_conclusion(conclusion: SurvivalConclusion) -> Optional[_KaplanMeierEstimatorModel]:
        """Build pydantic model from SurvivalConclusion

        Args:
            conclusion (SurvivalConclusion): conclusion

        Returns:
            Optional[_KaplanMeierEstimatorModel]: pydantic model or None if
            estimator of the conclusion is None
        """
        return (
            _KaplanMeierEstimatorModel(
                **conclusion.estimator.get_dict(),
                value=conclusion.value
            )
            if conclusion.estimator is not None
            else None
        )

    def to_estimator_object(self) -> KaplanMeierEstimator:
        """Convert pydantic model to KaplanMeierEstimator object

        Returns:
            KaplanMeierEstimator: estimator object
        """
        return KaplanMeierEstimator().update(
            self._to_kaplan_meier_estimator_dict(),
            update_additional_indicators=True
        )

    def _to_kaplan_meier_estimator_dict(self) -> KaplanMeierEstimatorDict:
        value: dict = self.model_dump()
        del value["value"]
        return value
