"""
Contains classes for regression ruleset JSON serialization.
"""
from __future__ import annotations

from typing import Optional

from decision_rules.core.coverage import Coverage
from decision_rules.regression.rule import RegressionConclusion
from decision_rules.regression.rule import RegressionRule
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.serialization._regression.rule import \
    _RegressionRuleSerializer
from decision_rules.serialization.utils import JSONClassSerializer
from decision_rules.serialization.utils import JSONSerializer
from decision_rules.serialization.utils import register_serializer
from pydantic import BaseModel


class _RegressionMetaDataModel(BaseModel):
    attributes: list[str]
    decision_attribute: str
    y_train_median: float


@register_serializer(RegressionRuleSet)
class _RegressionRuleSetSerializer(JSONClassSerializer):

    class _Model(BaseModel):
        meta: Optional[_RegressionMetaDataModel]
        rules: list[_RegressionRuleSerializer._Model]

    @classmethod
    def _from_pydantic_model(cls: type, model: _Model) -> RegressionRuleSet:
        ruleset = RegressionRuleSet(
            rules=[
                JSONSerializer.deserialize(
                    rule,
                    RegressionRule
                ) for rule in model.rules
            ],

        )
        ruleset.column_names = model.meta.attributes
        ruleset.decision_attribute = model.meta.decision_attribute
        for i, rule in enumerate(ruleset.rules):
            rule.column_names = ruleset.column_names
            if rule.coverage is None:
                rule.coverage = Coverage(None, None, None, None)
            rule.column_names = ruleset.column_names
            rule.conclusion.column_name = model.meta.decision_attribute
            rule.train_covered_y_mean = rule.conclusion.train_covered_y_mean
            if model.rules[i].coverage is not None:
                rule.coverage = Coverage(
                    **model.rules[i].coverage.model_dump())
        ruleset._y_train_median = model.meta.y_train_median  # pylint: disable=protected-access
        ruleset.default_conclusion = RegressionConclusion(
            value=model.meta.y_train_median,
            low=model.meta.y_train_median,
            high=model.meta.y_train_median,
            column_name=model.meta.decision_attribute,
        )
        return ruleset

    @classmethod
    def _to_pydantic_model(cls: type, instance: RegressionRuleSet) -> _Model:
        if len(instance.rules) == 0:
            raise ValueError('Cannot serialize empty ruleset.')
        return _RegressionRuleSetSerializer._Model(
            meta=_RegressionMetaDataModel(
                attributes=instance.column_names,
                decision_attribute=instance.rules[0].conclusion.column_name,
                y_train_median=instance.y_train_median
            ),
            rules=[
                JSONSerializer.serialize(rule) for rule in instance.rules
            ]
        )
