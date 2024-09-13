"""
Contains classes for regression ruleset JSON serialization.
"""
from __future__ import annotations

from typing import Optional

from decision_rules.core.coverage import Coverage
from decision_rules.serialization._survival.rule import _SurvivalRuleSerializer
from decision_rules.serialization.utils import JSONClassSerializer
from decision_rules.serialization.utils import JSONSerializer
from decision_rules.serialization.utils import register_serializer
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator
from decision_rules.survival.rule import SurvivalRule
from decision_rules.survival.ruleset import SurvivalRuleSet
from pydantic import BaseModel


class _SurvivalMetaDataModel(BaseModel):
    attributes: list[str]
    decision_attribute: str
    survival_time_attribute: str
    default_conclusion: dict[str, list[float]]


@register_serializer(SurvivalRuleSet)
class _SurvivalRuleSetSerializer(JSONClassSerializer):

    class _Model(BaseModel):
        meta: Optional[_SurvivalMetaDataModel]
        rules: list[_SurvivalRuleSerializer._Model]

    @classmethod
    def _from_pydantic_model(cls: type, model: _Model) -> SurvivalRuleSet:
        ruleset = SurvivalRuleSet(
            rules=[
                JSONSerializer.deserialize(
                    rule,
                    SurvivalRule
                ) for rule in model.rules
            ],
            survival_time_attr=model.meta.survival_time_attribute
        )
        ruleset.column_names = model.meta.attributes
        ruleset.decision_attribute = model.meta.decision_attribute
        ruleset.default_conclusion.estimator = KaplanMeierEstimator().update(
            model.meta.default_conclusion, update_additional_indicators=True)
        for i, rule in enumerate(ruleset.rules):
            rule.column_names = ruleset.column_names
            if rule.coverage is None:
                rule.coverage = Coverage(None, None, None, None)
            rule.set_survival_time_attr(model.meta.survival_time_attribute)
            rule.conclusion.column_name = model.meta.decision_attribute
            rule.conclusion.value = rule.conclusion.value
            rule.conclusion.median_survival_time_ci_lower = rule.conclusion.median_survival_time_ci_lower,
            rule.conclusion.median_survival_time_ci_upper = rule.conclusion.median_survival_time_ci_upper,
            if model.rules[i].coverage is not None:
                rule.coverage = Coverage(
                    **model.rules[i].coverage.model_dump())
        return ruleset

    @classmethod
    def _to_pydantic_model(cls: type, instance: SurvivalRuleSet) -> _Model:
        if len(instance.rules) == 0:
            raise ValueError('Cannot serialize empty ruleset.')
        return _SurvivalRuleSetSerializer._Model(
            meta=_SurvivalMetaDataModel(
                attributes=instance.column_names,
                decision_attribute=instance.rules[0].conclusion.column_name,
                survival_time_attribute=instance.rules[0].survival_time_attr,
                default_conclusion=instance.default_conclusion.estimator.get_dict()
            ),
            rules=[
                JSONSerializer.serialize(rule) for rule in instance.rules
            ]
        )
