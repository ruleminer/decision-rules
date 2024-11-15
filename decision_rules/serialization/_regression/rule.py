"""
Contains classes for regression rule's JSON serialization.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from decision_rules.regression.rule import RegressionConclusion, RegressionRule
from decision_rules.serialization._core.rule import _BaseRuleSerializer
from decision_rules.serialization.utils import (JSONClassSerializer,
                                                SerializationModes,
                                                register_serializer)


@register_serializer(RegressionConclusion)
class _RegressionRuleConclusionSerializer(JSONClassSerializer):

    class _Model(BaseModel):
        low: float
        value: Any
        high: float
        fixed: bool
        train_covered_y_mean: Optional[float] = None
        train_covered_y_std: Optional[float] = None
        train_covered_y_min: Optional[float] = None
        train_covered_y_max: Optional[float] = None

    @classmethod
    def _from_pydantic_model(cls: type, model: _Model) -> RegressionConclusion:
        conclusion = RegressionConclusion(
            value=model.value,
            column_name=None,
            low=model.low,
            high=model.high,
            fixed=model.fixed
        )
        conclusion.train_covered_y_mean = model.train_covered_y_mean
        conclusion.train_covered_y_std = model.train_covered_y_std
        conclusion.train_covered_y_min = model.train_covered_y_min
        conclusion.train_covered_y_max = model.train_covered_y_max
        return conclusion

    @classmethod
    def _to_pydantic_model(
        cls: type,
        instance: RegressionConclusion,
        mode: SerializationModes
    ) -> _Model:
        if mode == SerializationModes.FULL:
            train_covered_y_mean = instance.train_covered_y_mean
            train_covered_y_std = instance.train_covered_y_std
            train_covered_y_min = instance.train_covered_y_min
            train_covered_y_max = instance.train_covered_y_max
        else:
            train_covered_y_mean = train_covered_y_std = train_covered_y_min = \
                train_covered_y_max = None
        return _RegressionRuleConclusionSerializer._Model(
            value=instance.value,
            low=instance.low,
            high=instance.high,
            fixed=instance.fixed,
            train_covered_y_mean=train_covered_y_mean,
            train_covered_y_std=train_covered_y_std,
            train_covered_y_min=train_covered_y_min,
            train_covered_y_max=train_covered_y_max,
        )


@register_serializer(RegressionRule)
class _RegressionRuleSerializer(_BaseRuleSerializer):
    rule_class = RegressionRule
    conclusion_class = RegressionConclusion
