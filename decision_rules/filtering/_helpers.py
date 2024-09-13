from collections import defaultdict
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.core.coverage import Coverage
from decision_rules.core.rule import AbstractRule
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.survival.ruleset import SurvivalRuleSet
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error


"""
HELPER FUNCTIONS FOR FILTERING RULESETS

1.
Functions which calculate prediction score for each type of ruleset.
The following metrics are used for scoring:
- Classification - balanced accuracy,
- Regression - negative mean squared error,
- Survival - integrated Brier score.

2.
Common methods used in forward and backward algorithms

"""


def _calculate_classification_prediction_score(
        ruleset: AbstractRuleSet,
        _x: pd.DataFrame,
        y: pd.Series,
        coverage_matrix: np.array
) -> float:
    y_pred = ruleset.predict_using_coverage_matrix(coverage_matrix)
    return balanced_accuracy_score(y, y_pred)


def _calculate_regression_prediction_score(
        ruleset: AbstractRuleSet,
        _x: pd.DataFrame,
        y: pd.Series,
        coverage_matrix: np.array
) -> float:
    y_pred = ruleset.predict_using_coverage_matrix(coverage_matrix)
    return -mean_squared_error(y, y_pred)


def _calculate_survival_prediction_score(
        ruleset: SurvivalRuleSet,
        x: pd.DataFrame,
        y: pd.Series,
        _coverage_matrix: np.array = None
) -> float:
    return -ruleset.integrated_bier_score(x, y)


METRICS = {
    ClassificationRuleSet: _calculate_classification_prediction_score,
    RegressionRuleSet: _calculate_regression_prediction_score,
    SurvivalRuleSet: _calculate_survival_prediction_score,
}


def calculate_ruleset_prediction_score(
        ruleset: AbstractRuleSet,
        x: pd.DataFrame,
        y: pd.Series,
        coverage_matrix: np.array
) -> float:
    """Calculate prediction score for the given ruleset.

    Args:
        ruleset (AbstractRuleSet): ruleset to evaluate
        x (pd.DataFrame): dataframe with the independent variables
        y (pd.Series): series with the dependent variable
        coverage_matrix (np.array): matrix of covered examples

    Returns:
        float: prediction score
    """
    if not len(ruleset.rules):
        return float("-inf")
    return METRICS[type(ruleset)](ruleset, x, y, coverage_matrix)


def split_and_sort_ruleset(
        ruleset: AbstractRuleSet,
        x: pd.DataFrame, y: pd.Series,
        measure: Optional[Callable[[Coverage], float]],
        ascending: bool = True
) -> list[AbstractRule]:
    """Prepare ruleset for filtering by splitting it into sub-rulesets and sorting them by their quality.

    Args:
        ruleset (AbstractRuleSet): original ruleset
        x (pd.DataFrame): dataframe with the independent variables
        y (pd.Series): series with the dependent variable
        measure (Optional[Callable[[Coverage], float]]): voting measure using for sorting rules
        ascending (bool): whether to sort rules in ascending or descending order of quality

    Returns:
        list[AbstractRule]: list of rules sorted by their quality and interlaced between classes (for classification)
    """
    sort_operator = 1 if ascending else -1

    if isinstance(ruleset, ClassificationRuleSet):
        # group rules by their conclusion class
        new_rules = defaultdict(list)
        for rule in ruleset.rules:
            new_rules[rule.conclusion].append(rule)
        # sort sub-rulesets by their predictive quality
        new_rules = list(new_rules.values())
        prediction_scores = []
        for sub_ruleset in new_rules:
            sub_ruleset = ClassificationRuleSet(sub_ruleset)
            covered_matrix = sub_ruleset.update(x, y, measure)
            score = calculate_ruleset_prediction_score(
                sub_ruleset, x, y, covered_matrix)
            prediction_scores.append(score)
        new_rules = [new_rules[i]
                     for i in np.array(prediction_scores).argsort()[::sort_operator]]
    else:
        # for other problems simply group all rules in one list
        new_rules = [[rule for rule in ruleset.rules]]

    # sort rules within each sub-ruleset by their quality, expressed as voting weight
    sort_operator = 1 if ascending else -1
    for sub_ruleset in new_rules:
        sub_ruleset.sort(key=lambda r: sort_operator * r.voting_weight)

    # interlace them all in one list in the correct order
    all_rules = []
    while any(new_rules):
        for sub_ruleset in new_rules:
            if sub_ruleset:
                all_rules.append(sub_ruleset.pop())

    return all_rules
