from typing import Any

import pandas as pd

import decision_rules.ruleset_factories._factories as factories
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.ruleset_factories._factories.abstract_factory import \
    AbstractFactory


def ruleset_factory(
    model: Any, X_train: pd.DataFrame, y_train: pd.Series
) -> AbstractRuleSet:
    """Creates editable ruleset model from rule-based model
    from various ML packages.

    Args:
        model (Any): rule-based model instance
        X_train (pd.DataFrame):
        y_train (pd.Series):

    Raises:
        ValueError: if no factory exist for given model type

    Returns:
        AbstractRuleSet: editable ruleset model, either classification
            ruleset or regression or survival one.
    """
    # get class name
    model_class_name: str = model.__class__.__name__
    # get also names of parent classes (in case it is an expert induction model)
    model_parent_names: list[str] = [cls.__name__ for cls in model.__class__.mro()]
    class_names = [model_class_name] + model_parent_names
    if "RuleClassifier" in class_names:
        factory: AbstractFactory = (
            factories.classification.get_rulekit_factory_class()()
        )
    elif "RuleRegressor" in class_names:
        factory: AbstractFactory = factories.regression.get_rulekit_factory_class()()
    elif "SurvivalRules" in class_names:
        factory: AbstractFactory = factories.survival.get_rulekit_factory_class()()
    else:
        raise ValueError(f'No ruleset factory class for type: "{model_class_name}"')
    return factory.make(model, X_train, y_train)
