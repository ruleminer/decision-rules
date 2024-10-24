from typing import TypedDict

import numpy as np
import pandas as pd
from decision_rules.problem import ProblemTypes
from decision_rules.survival import SurvivalRuleSet


class SurvivalGeneralPredictionIndicators(TypedDict):
    ibs: float
    covered_by_prediction: int
    not_covered_by_prediction: int


class SurvivalPredictionIndicators(TypedDict):
    type_of_problem: str
    general: SurvivalGeneralPredictionIndicators


def _drop_uncovered_examples(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    EMPTY_VALUE: object = None
    covered_examples_mask: np.ndarray = y_pred != EMPTY_VALUE
    return y_true[covered_examples_mask], y_pred[covered_examples_mask]


def calculate_for_survival(
    ruleset: SurvivalRuleSet,
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    calculate_only_for_covered_examples: bool = False
) -> SurvivalRuleSet:
    """Calculate prediction indicators for survival problem.

    Args:
        ruleset (SurvivalRuleSet): ruleset
        X (pd.DataFrame): Dataset
        y_true (np.ndarray): Survival status column
        y_pred (np.ndarray): Array containing the predicted class labels.
        calculate_only_for_covered_examples (bool, optional): If true, it will
            calculate indicators only for the examples where prediction was not
            empty. Otherwise, it will calculate indicators for all the examples.
            Defaults to False.

    Returns:
        SurvivalPredictionIndicators:  A dictionary containing
        prediction indicators
    """
    all_examples = len(y_true)
    if calculate_only_for_covered_examples:
        y_true, y_pred = _drop_uncovered_examples(y_true, y_pred)
    covered_by_prediction = len(y_true)
    not_covered_by_prediction = all_examples - covered_by_prediction

    return SurvivalPredictionIndicators(
        type_of_problem=ProblemTypes.SURVIVAL.value,
        general=SurvivalGeneralPredictionIndicators(
            ibs=ruleset.integrated_bier_score(X, y_true, y_pred),
            covered_by_prediction=covered_by_prediction,
            not_covered_by_prediction=not_covered_by_prediction,
        )
    )
