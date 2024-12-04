"""
Contains classes for regression ruleset JSON serialization.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from decision_rules.core.coverage import Coverage
from decision_rules.regression.rule import RegressionConclusion, RegressionRule
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.serialization._regression.rule import (
    _RegressionRuleConclusionSerializer, _RegressionRuleSerializer)
from decision_rules.serialization.utils import (JSONClassSerializer,
                                                JSONSerializer,
                                                SerializationModes,
                                                register_serializer)


class _RegressionMetaDataModel(BaseModel):
    attributes: list[str]
    decision_attribute: str
    y_train_median: float
    default_conclusion: Optional[_RegressionRuleConclusionSerializer._Model] = None


@register_serializer(RegressionRuleSet)
class _RegressionRuleSetSerializer(JSONClassSerializer):

    class _Model(BaseModel):
        meta: _RegressionMetaDataModel
        rules: list[_RegressionRuleSerializer._Model]

    @classmethod
    def _from_pydantic_model(cls: type, model: _Model) -> RegressionRuleSet:
        ruleset = RegressionRuleSet(  # pylint: disable=abstract-class-instantiated
            rules=[
                JSONSerializer.deserialize(rule, RegressionRule) for rule in model.rules
            ],
        )
        ruleset.column_names = model.meta.attributes
        ruleset.decision_attribute = model.meta.decision_attribute
        for i, rule in enumerate(ruleset.rules):
            rule.column_names = ruleset.column_names
            rule.column_names = ruleset.column_names
            rule.conclusion.column_name = model.meta.decision_attribute
            rule.train_covered_y_mean = rule.conclusion.train_covered_y_mean
            if model.rules[i].coverage is not None:
                rule.coverage = Coverage(**model.rules[i].coverage.model_dump())
        ruleset._y_train_median = (
            model.meta.y_train_median
        )  # pylint: disable=protected-access

        _RegressionRuleSetSerializer._set_default_conclusion(ruleset, model)
        return ruleset

    @classmethod
    def _set_default_conclusion(cls: type, ruleset: RegressionRuleSet, model: _Model):
        if model.meta.default_conclusion is not None:
            default_conclusion: RegressionConclusion = JSONSerializer.deserialize(
                model.meta.default_conclusion, target_class=RegressionConclusion
            )
            ruleset.default_conclusion = default_conclusion
        else:
            ruleset.default_conclusion = RegressionConclusion(
                value=model.meta.y_train_median,
                low=model.meta.y_train_median,
                high=model.meta.y_train_median,
                column_name=model.meta.decision_attribute,
            )

    @classmethod
    def _to_pydantic_model(
        cls: type, instance: RegressionRuleSet, mode: SerializationModes
    ) -> _Model:
        if len(instance.rules) == 0:
            raise ValueError("Cannot serialize empty ruleset.")
        if mode == SerializationModes.FULL:
            default_conclusion = JSONSerializer.serialize(
                instance.default_conclusion, mode
            )
        else:
            default_conclusion = None
        return _RegressionRuleSetSerializer._Model(
            meta=_RegressionMetaDataModel(
                attributes=instance.column_names,
                decision_attribute=instance.rules[0].conclusion.column_name,
                y_train_median=instance.y_train_median,
                default_conclusion=default_conclusion,
            ),
            rules=[JSONSerializer.serialize(rule, mode) for rule in instance.rules],
        )
