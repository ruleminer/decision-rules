from copy import deepcopy
from typing import Callable
from typing import Optional

import pandas as pd
from decision_rules.core.coverage import Coverage
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.filtering._helpers import calculate_ruleset_prediction_score
from decision_rules.filtering._helpers import split_and_sort_ruleset


def filter_ruleset_with_forward(
        ruleset: AbstractRuleSet,
        X: pd.DataFrame,
        y: pd.Series,
        loss: float,
        measure: Optional[Callable[[Coverage], float]],
) -> AbstractRuleSet:
    """Filter ruleset using forward algorithm.

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
        filtered_ruleset, X, y, measure, ascending=True)
    filtered_ruleset.rules = new_rules
    coverage_matrix = filtered_ruleset.update(X, y, measure)
    filtered_ruleset.rules = []
    filtered_ruleset_score = float("-inf")

    # implement forward algorithm
    added_rules = []
    for i, rule in enumerate(new_rules):
        # add a new rule and its index to lists
        filtered_ruleset.rules.append(rule)
        added_rules.append(i)
        # select a subset of the coverage matrix corresponding to rules added so far
        new_coverage_matrix = coverage_matrix[:, added_rules]
        new_ruleset_score = calculate_ruleset_prediction_score(
            filtered_ruleset, X, y, new_coverage_matrix)
        # if the score is better, we keep the rule and update the score
        if new_ruleset_score > filtered_ruleset_score:
            filtered_ruleset_score = new_ruleset_score
        # if not, we remove the rule again
        else:
            filtered_ruleset.rules.pop()
            added_rules.pop()
        # if the score is not worse than the original ruleset score, we stop
        if new_ruleset_score >= target_score:
            break

    filtered_ruleset.update(X, y, measure)

    return filtered_ruleset
