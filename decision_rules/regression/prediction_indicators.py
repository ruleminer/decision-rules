import math
from typing import TypedDict

import numpy as np
from decision_rules.problem import ProblemTypes
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# funky definition because of the field with invalid identifier name "R^2"
RegressionGeneralPredictionIndicators = TypedDict(
    'RegressionGeneralPredictionIndicators',
    {
        'RMSE': float,
        'MAE': float,
        'MAPE': float,
        'rRMSE': float,
        'rMAE': float,
        'maxError': float,
        'R^2': float,
        'Covered_by_prediction': int,
        'Not_covered_by_prediction': int,
    }
)


class RegressionPredictionHistogram(TypedDict):
    max: float
    min: float
    bin_edges: list[float]
    histogram: list[int]


class RegressionPredictionIndicators(TypedDict):
    type_of_problem: str
    general: RegressionGeneralPredictionIndicators
    histogram: RegressionPredictionHistogram


def _drop_uncovered_examples(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covered_examples_mask: np.ndarray = ~np.isnan(y_pred)
    return y_true[covered_examples_mask], y_pred[covered_examples_mask]


def calculate_for_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    calculate_only_for_covered_examples: bool = False
) -> RegressionPredictionIndicators:
    """Calculate prediction indicators for regression problem.

    Args:
        y_true (np.ndarray): Array containing the actual labels.
        y_pred (np.ndarray): Array containing the predicted labels.
        calculate_only_for_covered_examples (bool, optional): If true, it will
            calculate indicators only for the examples where prediction was not
            empty. Otherwise, it will calculate indicators for all the examples.
            Defaults to False.

    Returns:
        RegressionPredictionIndicators:  A dictionary containing
        prediction indicators
    """
    all_examples = len(y_true)
    if calculate_only_for_covered_examples:
        y_true, y_pred = _drop_uncovered_examples(y_true, y_pred)
    covered_by_prediction = len(y_true)
    not_covered_by_prediction = all_examples - covered_by_prediction

    RMSE = math.sqrt(mean_squared_error(y_true, y_pred))
    MAE = mean_absolute_error(y_true, y_pred)
    MAPE = mean_absolute_percentage_error(y_true, y_pred)
    rRMSE = RMSE / np.mean(y_true)
    rMAE = MAE / np.mean(y_true)
    maxError = max_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)

    errors = np.array(y_pred) - np.array(y_true)
    bin_edges = np.histogram_bin_edges(errors, bins='auto')
    histogram, _ = np.histogram(errors, bins=bin_edges)
    return RegressionPredictionIndicators(
        type_of_problem=ProblemTypes.REGRESSION.value,
        general=RegressionGeneralPredictionIndicators(**{
            "RMSE": RMSE,
            "MAE": MAE,
            "MAPE": MAPE,
            "rRMSE": rRMSE,
            "rMAE": rMAE,
            "maxError": maxError,
            "R^2": R2,
            "Covered_by_prediction": covered_by_prediction,
            "Not_covered_by_prediction": not_covered_by_prediction,
        }),
        histogram=RegressionPredictionHistogram(
            max=max(errors),
            min=min(errors),
            bin_edges=bin_edges.tolist(),
            histogram=histogram.tolist()
        )
    )
