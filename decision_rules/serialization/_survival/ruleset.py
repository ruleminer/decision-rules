"""
Contains classes for regression ruleset JSON serialization.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from decision_rules.core.coverage import Coverage
from decision_rules.serialization._survival.kaplan_meier import \
    _KaplanMeierEstimatorModel
from decision_rules.serialization._survival.rule import _SurvivalRuleSerializer
from decision_rules.serialization.utils import (JSONClassSerializer,
                                                JSONSerializer,
                                                register_serializer)
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator
from decision_rules.survival.rule import SurvivalRule
from decision_rules.survival.ruleset import SurvivalRuleSet


class _SurvivalMetaDataModel(BaseModel):
    attributes: list[str]
    decision_attribute: str
    survival_time_attribute: str
    default_conclusion: _KaplanMeierEstimatorModel


@register_serializer(SurvivalRuleSet)
class _SurvivalRuleSetSerializer(JSONClassSerializer):

    class _Model(BaseModel):
        meta: Optional[_SurvivalMetaDataModel]
        rules: list[_SurvivalRuleSerializer._Model]

    @classmethod
    def _from_pydantic_model(cls: type, model: _Model) -> SurvivalRuleSet:
        ruleset = SurvivalRuleSet(  # pylint: disable=abstract-class-instantiated
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
        _SurvivalRuleSetSerializer._update_default_conclusion(
            ruleset, model
        )
        for i, rule in enumerate(ruleset.rules):
            rule.column_names = ruleset.column_names
            rule.set_survival_time_attr(model.meta.survival_time_attribute)
            rule.conclusion.column_name = model.meta.decision_attribute
            rule.conclusion.value = rule.conclusion.value
            rule.conclusion.median_survival_time_ci_lower = rule.conclusion.median_survival_time_ci_lower,
            rule.conclusion.median_survival_time_ci_upper = rule.conclusion.median_survival_time_ci_upper,
            if model.rules[i].coverage is not None:
                rule.coverage = Coverage(
                    **model.rules[i].coverage.model_dump()
                )
            else:
                rule.coverage = Coverage(None, None, None, None)
        return ruleset

    @classmethod
    def _update_default_conclusion(
        cls: type,
        ruleset: SurvivalRuleSet,
        model: _Model
    ):
        estimator: KaplanMeierEstimator = model.meta.default_conclusion. \
            to_estimator_object()
        ruleset.default_conclusion.estimator = estimator
        ruleset.default_conclusion.value = (
            estimator.median_survival_time
            if model.meta.default_conclusion.value is None
            else model.meta.default_conclusion.value
        )

    @classmethod
    def _to_pydantic_model(cls: type, instance: SurvivalRuleSet) -> _Model:
        if len(instance.rules) == 0:
            raise ValueError('Cannot serialize empty ruleset.')
        if instance.default_conclusion is None:
            default_conclusion = None
        else:
            default_conclusion = _KaplanMeierEstimatorModel.from_rule_conclusion(
                instance.default_conclusion
            )
        return _SurvivalRuleSetSerializer._Model(
            meta=_SurvivalMetaDataModel(
                attributes=instance.column_names,
                decision_attribute=instance.rules[0].conclusion.column_name,
                survival_time_attribute=instance.rules[0].survival_time_attr,
                default_conclusion=default_conclusion
            ),
            rules=[
                JSONSerializer.serialize(rule) for rule in instance.rules
            ]
        )
