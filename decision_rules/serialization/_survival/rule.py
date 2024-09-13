"""
Contains classes for survival rule's JSON serialization.
"""
from __future__ import annotations

from typing import Any

from decision_rules.serialization._core.rule import _BaseRuleSerializer
from decision_rules.serialization.utils import JSONClassSerializer
from decision_rules.serialization.utils import register_serializer
from decision_rules.survival.rule import SurvivalConclusion
from decision_rules.survival.rule import SurvivalRule
from pydantic import BaseModel


@register_serializer(SurvivalConclusion)
class _SurvivalRuleConclusionSerializer(JSONClassSerializer):

    class _Model(BaseModel):
        value: Any
        median_survival_time_ci_lower: Any
        median_survival_time_ci_upper: Any

    @classmethod
    def _from_pydantic_model(cls: type, model: _Model) -> SurvivalConclusion:
        conclusion = SurvivalConclusion(
            value=model.value,
            column_name=None
        )
        conclusion.median_survival_time_ci_lower = model.median_survival_time_ci_lower
        conclusion.median_survival_time_ci_upper = model.median_survival_time_ci_upper
        return conclusion

    @classmethod
    def _to_pydantic_model(
        cls: type,
        instance: SurvivalConclusion
    ) -> _Model:
        return _SurvivalRuleConclusionSerializer._Model(
            value=instance.value,
            median_survival_time_ci_lower=instance.median_survival_time_ci_lower,
            median_survival_time_ci_upper=instance.median_survival_time_ci_upper,
            column_name=instance.column_name,
        )


@register_serializer(SurvivalRule)
class _SurvivalRuleSerializer(_BaseRuleSerializer):
    rule_class = SurvivalRule
    conclusion_class = SurvivalConclusion
