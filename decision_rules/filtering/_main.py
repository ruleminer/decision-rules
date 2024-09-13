from enum import Enum
from typing import Callable
from typing import Optional

import pandas as pd
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.filtering._backward import filter_ruleset_with_backward
from decision_rules.filtering._coverage import filter_ruleset_with_coverage
from decision_rules.filtering._forward import filter_ruleset_with_forward
from decision_rules.helpers import get_measure_function_by_name
from decision_rules.survival.ruleset import SurvivalRuleSet


class FilterAlgorithm(str, Enum):
    Coverage = "coverage"
    Forward = "forward"
    Backward = "backward"


FILTER_ALGORITHM_MAPPING = {
    FilterAlgorithm.Coverage: filter_ruleset_with_coverage,
    FilterAlgorithm.Forward: filter_ruleset_with_forward,
    FilterAlgorithm.Backward: filter_ruleset_with_backward,
}


def filter_ruleset(
        ruleset: AbstractRuleSet,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: FilterAlgorithm,
        loss: Optional[float],
        measure: Optional[Callable or str] = None,
) -> AbstractRuleSet:
    """Filter ruleset using specified algorithm.

    Args:
        ruleset (AbstractRuleSet): ruleset to filter
        X (pd.DataFrame): dataset features
        y (pd.Series): dataset target
        algorithm (FilterAlgorithm): filtering algorithm to use
        loss (float): accepted loss of prediction quality (fraction)
        measure (Optional[Callable or str]): rule quality measure (voting measure) - a callable or a string (name)

    Returns:
        AbstractRuleSet: filtered ruleset
    """
    if loss is None:
        loss = 1.0
    if measure is None and not isinstance(ruleset, SurvivalRuleSet):
        raise ValueError(
            "Voting measure must be specified for classification and regression rulesets.")
    if measure is not None and isinstance(measure, str):
        measure = get_measure_function_by_name(measure)
    return FILTER_ALGORITHM_MAPPING[algorithm](ruleset, X, y, loss, measure)
