from copy import deepcopy
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd
from decision_rules.core.coverage import Coverage
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.filtering._helpers import calculate_ruleset_prediction_score
from decision_rules.filtering._helpers import split_and_sort_ruleset


def filter_ruleset_with_backward(
        ruleset: AbstractRuleSet,
        X: pd.DataFrame,
        y: pd.Series,
        loss: float,
        measure: Optional[Callable[[Coverage], float]],
) -> AbstractRuleSet:
    """Filter ruleset using backward algorithm.

    Args:
        ruleset (AbstractRuleSet): ruleset to filter
        X (pd.DataFrame): dataset features
        y (pd.Series): dataset target
        loss (float): accepted loss of prediction quality (fraction)
        measure (Optional[Callable[[Coverage], float]]): rule quality measure (voting measure)

    Returns:
        AbstractRuleSet: filtered ruleset
    """
    # get original ruleset score and calculate target score
    original_coverage_matrix = ruleset.update(X, y, measure)
    original_ruleset_score = calculate_ruleset_prediction_score(
        ruleset, X, y, original_coverage_matrix)
    if original_ruleset_score >= 0:
        target_score = original_ruleset_score * (1 - loss)
    else:
        target_score = original_ruleset_score * (1 + loss)

    # copy the original ruleset to create the new target filtered ruleset
    filtered_ruleset = deepcopy(ruleset)
    new_rules = split_and_sort_ruleset(
        filtered_ruleset, X, y, measure, ascending=False)
    filtered_ruleset.rules = new_rules
    coverage_matrix = filtered_ruleset.update(X, y, measure)

    # implement backward algorithm
    # iterate over an index at which we remove the next rule
    i = 0
    while i < len(filtered_ruleset.rules) and len(filtered_ruleset.rules) > 1:
        # we remove the rule from the ruleset and its corresponding column from the coverage matrix
        deleted_rule = filtered_ruleset.rules.pop(i)
        new_coverage_matrix = np.delete(coverage_matrix, i, axis=1)
        new_ruleset_score = calculate_ruleset_prediction_score(
            filtered_ruleset, X, y, new_coverage_matrix)
        # if score is not worse than the target score,
        # we update it, keep the rule removed, and continue
        if new_ruleset_score >= target_score:
            coverage_matrix = new_coverage_matrix
        # if not, we restore the last removed rule and move to the next index
        else:
            filtered_ruleset.rules.insert(i, deleted_rule)
            i += 1

    filtered_ruleset.update(X, y, measure)

    return filtered_ruleset
