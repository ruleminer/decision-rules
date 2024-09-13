"""
Contains classes for classification rule's JSON serialization.
"""
from __future__ import annotations

from typing import Any

from decision_rules.classification.rule import ClassificationConclusion
from decision_rules.classification.rule import ClassificationRule
from decision_rules.serialization._core.rule import _BaseRuleSerializer
from decision_rules.serialization.utils import JSONClassSerializer
from decision_rules.serialization.utils import register_serializer
from pydantic import BaseModel


@register_serializer(ClassificationConclusion)
class _ClassificationRuleConclusionSerializer(JSONClassSerializer):

    class _Model(BaseModel):
        value: Any

    @classmethod
    def _from_pydantic_model(cls: type, model: _Model) -> ClassificationConclusion:
        return ClassificationConclusion(
            value=model.value,
            column_name=None
        )

    @classmethod
    def _to_pydantic_model(
        cls: type,
        instance: ClassificationConclusion
    ) -> _Model:
        return _ClassificationRuleConclusionSerializer._Model(
            value=instance.value,
        )


@register_serializer(ClassificationRule)
class _ClassificationRuleSerializer(_BaseRuleSerializer):
    rule_class = ClassificationRule
    conclusion_class = ClassificationConclusion
