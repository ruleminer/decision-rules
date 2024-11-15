"""
Contains common classes for rules JSON serialization.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from decision_rules.core.coverage import Coverage
from decision_rules.core.rule import AbstractRule
from decision_rules.serialization._core.conditions import _ConditionSerializer
from decision_rules.serialization.utils import (JSONClassSerializer,
                                                JSONSerializer,
                                                register_serializer)


@register_serializer(Coverage)
class _CoverageSerializer(JSONClassSerializer):

    class _Model(BaseModel):
        p: Optional[int]
        n: Optional[int]
        P: Optional[int] = None
        N: Optional[int] = None

    @classmethod
    def _from_pydantic_model(cls: type, model: _Model) -> Coverage:
        return Coverage(
            p=int(model.p),
            n=int(model.n),
            P=int(model.P) if model.P is not None else None,
            N=int(model.N) if model.N is not None else None
        )

    @classmethod
    def _to_pydantic_model(
        cls: type,
        instance: Coverage
    ) -> _Model:
        return _CoverageSerializer._Model(
            p=int(instance.p) if instance.p is not None else None,
            n=int(instance.n) if instance.n is not None else None,
            P=int(instance.P) if instance.P is not None else None,
            N=int(instance.N) if instance.N is not None else None,
        )


class _BaseRuleSerializer(JSONClassSerializer):

    rule_class: type
    conclusion_class: type

    class _Model(BaseModel):
        uuid: str
        string: str
        premise: Any
        conclusion: Any
        coverage: Optional[_CoverageSerializer._Model] = None
        voting_weight: Optional[float] = None

    @classmethod
    def _from_pydantic_model(cls: type, model: _Model) -> AbstractRule:
        rule: AbstractRule = cls.rule_class(
            premise=_ConditionSerializer.deserialize(
                model.premise),
            conclusion=JSONSerializer.deserialize(
                model.conclusion,
                cls.conclusion_class
            ),
            column_names=[],  # must be populated when deserializing ruleset!
        )
        rule._uuid = model.uuid  # pylint: disable=protected-access
        if model.coverage is not None:
            rule.coverage = JSONSerializer.deserialize(
                model.coverage,
                Coverage
            )
        if model.voting_weight is not None:
            rule.voting_weight = model.voting_weight
        return rule

    @classmethod
    def _to_pydantic_model(cls: type, instance: AbstractRule) -> _Model:
        model = _BaseRuleSerializer._Model(
            uuid=instance.uuid,
            string=instance.__str__(  # pylint: disable=unnecessary-dunder-call
                show_coverage=False
            ),
            premise=JSONSerializer.serialize(
                instance.premise),  # pylint: disable=duplicate-code
            conclusion=JSONSerializer.serialize(instance.conclusion),
            coverage=JSONSerializer.serialize(instance.coverage),
            voting_weight=instance.voting_weight,
        )
        return model
