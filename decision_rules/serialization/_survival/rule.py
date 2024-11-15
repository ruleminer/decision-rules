"""
Contains classes for survival rule's JSON serialization.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from decision_rules.serialization._core.rule import _BaseRuleSerializer
from decision_rules.serialization._survival.kaplan_meier import \
    _KaplanMeierEstimatorModel
from decision_rules.serialization.utils import (JSONClassSerializer,
                                                SerializationModes,
                                                register_serializer)
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator
from decision_rules.survival.rule import SurvivalConclusion, SurvivalRule


@register_serializer(SurvivalConclusion)
class _SurvivalRuleConclusionSerializer(JSONClassSerializer):

    class _Model(BaseModel):
        value: Any
        median_survival_time_ci_lower: Any
        median_survival_time_ci_upper: Any
        estimator: Optional[_KaplanMeierEstimatorModel] = None

    @classmethod
    def _from_pydantic_model(cls: type, model: _Model) -> SurvivalConclusion:
        conclusion = SurvivalConclusion(value=model.value, column_name=None)
        conclusion.median_survival_time_ci_lower = model.median_survival_time_ci_lower
        conclusion.median_survival_time_ci_upper = model.median_survival_time_ci_upper
        if model.estimator is not None:
            conclusion.estimator = model.estimator.to_estimator_object()
            conclusion.value = model.estimator.value
        return conclusion

    @classmethod
    def _to_pydantic_model(
        cls: type, instance: SurvivalConclusion, mode: SerializationModes
    ) -> _Model:
        if mode == SerializationModes.FULL:
            estimator: KaplanMeierEstimator = (
                _KaplanMeierEstimatorModel.from_conclusion(instance)
            )
        else:
            estimator = None
        return _SurvivalRuleConclusionSerializer._Model(
            value=instance.value,
            median_survival_time_ci_lower=instance.median_survival_time_ci_lower,
            median_survival_time_ci_upper=instance.median_survival_time_ci_upper,
            column_name=instance.column_name,
            estimator=estimator,
        )


@register_serializer(SurvivalRule)
class _SurvivalRuleSerializer(_BaseRuleSerializer):
    rule_class = SurvivalRule
    conclusion_class = SurvivalConclusion
