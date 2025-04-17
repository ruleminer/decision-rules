"""
Contains classes for classification ruleset JSON serialization.
"""
from __future__ import annotations

from typing import Any
from typing import Optional

import numpy as np
from pydantic import BaseModel

from decision_rules.classification.rule import ClassificationRule
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.core.coverage import Coverage
from decision_rules.serialization._classification.rule import _ClassificationRuleConclusionSerializer
from decision_rules.serialization._classification.rule import _ClassificationRuleSerializer
from decision_rules.serialization.utils import JSONClassSerializer
from decision_rules.serialization.utils import JSONSerializer
from decision_rules.serialization.utils import register_serializer
from decision_rules.serialization.utils import SerializationModes


class _ClassificationMetaDataModel(BaseModel):
    attributes: list[str]
    decision_attribute: str
    decision_attribute_distribution: dict[Any, int]
    default_conclusion: Optional[_ClassificationRuleConclusionSerializer._Model] = None


@register_serializer(ClassificationRuleSet)
class _ClassificationRuleSetSerializer(JSONClassSerializer):

    class _Model(BaseModel):
        meta: _ClassificationMetaDataModel
        rules: list[_ClassificationRuleSerializer._Model]

    @classmethod
    def _from_pydantic_model(cls: type, model: _Model) -> ClassificationRuleSet:
        ruleset = ClassificationRuleSet(  # pylint: disable=abstract-class-instantiated
            rules=[
                JSONSerializer.deserialize(rule, ClassificationRule)
                for rule in model.rules
            ],
        )
        ruleset.y_values = np.array(
            list(model.meta.decision_attribute_distribution.keys())
        )
        ruleset.column_names = model.meta.attributes
        ruleset.decision_attribute = model.meta.decision_attribute
        _ClassificationRuleSetSerializer._calculate_P_N(model, ruleset)
        _ClassificationRuleSetSerializer._set_default_conclusion(
            ruleset, model.meta.default_conclusion
        )
        return ruleset

    @classmethod
    def _calculate_P_N(
        cls: type, model: _Model, ruleset: ClassificationRuleSet
    ):  # pylint: disable=invalid-name
        all_example_count = sum(
            model.meta.decision_attribute_distribution.values())
        ruleset.train_P = {}
        ruleset.train_N = {}
        for y_value, count in model.meta.decision_attribute_distribution.items():
            ruleset.train_P[y_value] = count
            ruleset.train_N[y_value] = all_example_count - count
        for rule in ruleset.rules:
            rule.column_names = ruleset.column_names
            if rule.coverage is not None:
                rule.coverage.P = ruleset.train_P[rule.conclusion.value]
                rule.coverage.N = ruleset.train_N[rule.conclusion.value]
            rule.conclusion.column_name = model.meta.decision_attribute

    @classmethod
    def _set_default_conclusion(
        cls: type,
        ruleset: ClassificationRuleSet,
        default_conclusion_serialized: _ClassificationRuleConclusionSerializer,
    ):
        if default_conclusion_serialized is not None:
            default_conclusion = _ClassificationRuleConclusionSerializer._from_pydantic_model(  # pylint: disable=protected-access
                default_conclusion_serialized
            )
            ruleset.default_conclusion = default_conclusion
        else:
            ruleset._update_majority_class()  # pylint: disable=protected-access

    @classmethod
    def _to_pydantic_model(
        cls: type, instance: ClassificationRuleSet, mode: SerializationModes
    ) -> _Model:
        if len(instance.rules) == 0:
            raise ValueError("Cannot serialize empty ruleset.")

        return _ClassificationRuleSetSerializer._Model(
            meta=_ClassificationMetaDataModel(
                attributes=instance.column_names,
                decision_attribute=instance.rules[0].conclusion.column_name,
                decision_attribute_distribution=dict(instance.train_P.items()),
                default_conclusion=JSONSerializer.serialize(
                    instance.default_conclusion, mode
                ),
            ),
            rules=[JSONSerializer.serialize(rule, mode)
                   for rule in instance.rules],
        )
