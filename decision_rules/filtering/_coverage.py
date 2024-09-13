from copy import deepcopy
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd
from decision_rules.core.coverage import Coverage
from decision_rules.core.ruleset import AbstractRuleSet


def filter_ruleset_with_coverage(
        ruleset: AbstractRuleSet,
        X: pd.DataFrame,
        y: pd.Series,
        _loss: float,
        measure: Optional[Callable[[Coverage], float]],
) -> AbstractRuleSet:
    """Filter ruleset using coverage algorithm.

    Args:
        ruleset (AbstractRuleSet): ruleset to filter
        X (pd.DataFrame): dataset features
        y (pd.Series): dataset target
        _loss (float): accepted loss of prediction quality (fraction) [only for compatibility]
        measure (Optional[Callable[[Coverage], float]]): rule quality measure (voting measure)

    Returns:
        AbstractRuleSet: filtered ruleset
    """
    X_orig = X.copy()
    y_orig = y.copy()

    # create target filtered ruleset by copying the original
    filtered_ruleset = deepcopy(ruleset)
    new_rules = []
    filtered_ruleset.rules = np.array(filtered_ruleset.rules)
    # we iterate until there are no rules left
    while len(filtered_ruleset.rules):
        # calculate coverage matrix for the current version of ruleset and dataset
        coverage_matrix = filtered_ruleset.update(X, y, measure).T
        # remove rules which do not cover anything anymore
        filtered_ruleset.rules = filtered_ruleset.rules[coverage_matrix.any(1)]
        coverage_matrix = coverage_matrix[coverage_matrix.any(1)]
        if not len(filtered_ruleset.rules):
            # it may happen that we also run out of rules here
            break
        # calculate quality of each rule
        rule_quality_scores = [
            rule.voting_weight for rule in filtered_ruleset.rules]
        # move the best-ranking rule from the original ruleset to the new rule list
        best_rule_idx = np.argmax(rule_quality_scores)
        best_rule = filtered_ruleset.rules[best_rule_idx]
        new_rules.append(best_rule)
        filtered_ruleset.rules = np.delete(
            filtered_ruleset.rules, best_rule_idx)
        # remove covered examples from the dataset
        best_rule_mask = coverage_matrix[best_rule_idx]
        X = X[~best_rule_mask]
        y = y[~best_rule_mask]
        # finish when all examples are covered
        if not len(X):
            break

    # prepare the new ruleset
    filtered_ruleset.rules = new_rules
    filtered_ruleset.update(X_orig, y_orig, measure)

    return filtered_ruleset
