from typing import Any
from typing import TypedDict
import warnings

import numpy as np
import math
from decision_rules.problem import ProblemTypes
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


class ClassificationGeneralPredictionIndicators(TypedDict):
    Balanced_accuracy: float
    Accuracy: float
    Cohen_kappa: float
    F1_micro: float
    F1_macro: float
    F1_weighted: float
    G_mean_micro: float
    G_mean_macro: float
    G_mean_weighted: float
    Recall_micro: float
    Recall_macro: float
    Recall_weighted: float
    Specificity: float
    Confusion_matrix: dict
    Covered_by_prediction: int
    Not_covered_by_prediction: int


class ClassificationPredictionIndicatorsForClass(TypedDict):
    TP: int
    FP: int
    TN: int
    FN: int
    Recall: float
    Specificity: float
    F1_score: float
    G_mean: float
    MCC: float
    PPV: float
    NPV: float
    LR_plus: float
    LR_minus: float
    Odd_ratio: float
    Relative_risk: float
    Confusion_matrix: dict


class ClassificationPredictionIndicators(TypedDict):
    type_of_problem: str
    general: ClassificationGeneralPredictionIndicators
    for_classes: dict[str, ClassificationPredictionIndicatorsForClass]


def _drop_uncovered_examples(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    EMPTY_VALUE: str = ''
    covered_examples_mask: np.ndarray = y_pred != EMPTY_VALUE
    return y_true[covered_examples_mask], y_pred[covered_examples_mask]


def calculate_for_classification(
    y_true: list[Any],
    y_pred: list[Any],
    calculate_only_for_covered_examples: bool = False
) -> ClassificationPredictionIndicators:
    """ Calculate prediction indicators for classification problem.

    Args:
        y_true (np.ndarray): Array containing the actual class labels.
        y_pred (np.ndarray): Array containing the predicted class labels.
        calculate_only_for_covered_examples (bool, optional): If true, it will
            calculate indicators only for the examples where prediction was not
            empty. Otherwise, it will calculate indicators for all the examples.
            Defaults to False.

    Returns:
        ClassificationPredictionIndicators: A dictionary containing various
        prediction indicators, including indicators for individual classes.
    """
    all_examples = len(y_true)
    if calculate_only_for_covered_examples:
        y_true, y_pred = _drop_uncovered_examples(y_true, y_pred)
    covered_by_prediction = len(y_true)
    not_covered_by_prediction = all_examples - covered_by_prediction

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.filterwarnings(
            action="always",
            category=UserWarning,
            message=r"y_pred contains classes not in y_true",
            module="sklearn.metrics"
        )
        balanced_accuracy: float = balanced_accuracy_score(
            y_true=y_true, y_pred=y_pred)
        accuracy: float = accuracy_score(
            y_true=y_true, y_pred=y_pred)
        kappa: float = cohen_kappa_score(y_true, y_pred)
        F1_macro: float = f1_score(y_true, y_pred, average='macro')
        F1_micro: float = f1_score(y_true, y_pred, average='micro')
        F1_weighted: float = f1_score(y_true, y_pred, average='weighted')
        G_mean_macro: float = geometric_mean_score(y_true, y_pred, average='macro')
        G_mean_micro: float = geometric_mean_score(y_true, y_pred, average='micro')
        G_mean_weighted: float = geometric_mean_score(
            y_true, y_pred, average='weighted')
        Recall_macro: float = recall_score(y_true, y_pred, average='macro', zero_division=0.0)
        Recall_micro: float = recall_score(y_true, y_pred, average='micro', zero_division=0.0)
        Recall_weighted: float = recall_score(y_true, y_pred, average='weighted', zero_division=0.0)
        c_matrix: np.ndarray = confusion_matrix(y_true, y_pred)

    for warning in caught_warnings:
        # modify msg to include the reason for the warning
        warnings.warn(
            message=(
                f"From sklearn.metrics: '{warning.message}'. This behavior could potentially "
                "result from the default conclusion being turned off during prediction."
            ),
            category=warning.category,
        )

    TN: int = c_matrix[0, 0]
    if c_matrix.shape[1] == 1:
        FP: int = 0
    else:
        FP: int = c_matrix[0, 1]

    specificity: float = TN / (TN + FP) if (TN + FP) != 0 else 0

    classes: list[Any] = np.unique(np.concatenate((y_true, y_pred))).tolist()
    general_confusion_matrix_dict: dict[str, Any] = {
        "classes": classes
    }
    class_indicators: dict[
        str,
        ClassificationPredictionIndicatorsForClass
    ] = {}
    for i, cls in enumerate(classes):
        class_data = calculate_class_indicators_classification(
            cls, y_true, y_pred)
        class_indicators[cls] = class_data
        general_confusion_matrix_dict[cls] = c_matrix[i, :].tolist()

    return ClassificationPredictionIndicators(
        type_of_problem=ProblemTypes.CLASSIFICATION.value,
        general=ClassificationGeneralPredictionIndicators(
            Balanced_accuracy=balanced_accuracy,
            Accuracy=accuracy,
            Cohen_kappa=kappa,
            F1_micro=F1_micro,
            F1_macro=F1_macro,
            F1_weighted=F1_weighted,
            G_mean_micro=G_mean_micro,
            G_mean_macro=G_mean_macro,
            G_mean_weighted=G_mean_weighted,
            Recall_micro=Recall_micro,
            Recall_macro=Recall_macro,
            Recall_weighted=Recall_weighted,
            Specificity=specificity,
            Confusion_matrix=general_confusion_matrix_dict,
            Covered_by_prediction=covered_by_prediction,
            Not_covered_by_prediction=not_covered_by_prediction,
        ),
        for_classes={
            cls: {key: value if isinstance(
                value, (float)) else value for key, value in class_data.items()}
            for cls, class_data in class_indicators.items()
        }
    )


def calculate_class_indicators_classification(
        cls: Any,
        y_true: list[Any],
        y_pred: list[Any]
) -> ClassificationPredictionIndicatorsForClass:
    """Calculate prediction indicators for a specific class.

    Args:
        cls (Any): The class for which to calculate the indicators.
        y_true (list[Any]): The true labels.
        y_pred (List[Any]): The predicted labels.

    Returns:
        ClassificationPredictionIndicatorsForClass: A dictionary representing
        the calculated prediction indicators for the class.
    """
    TP: int = int(np.count_nonzero(np.logical_and(y_true == cls, y_pred == cls)))
    FN: int = int(np.count_nonzero(np.logical_and(y_true == cls, y_pred != cls)))
    FP: int = int(np.count_nonzero(np.logical_and(y_true != cls, y_pred == cls)))
    TN: int = int(np.count_nonzero(np.logical_and(y_true != cls, y_pred != cls)))
    Precision: float = TP / (TP + FP) if (TP + FP) != 0 else 0
    Recall: float = TP / (TP + FN) if (TP + FN) != 0 else 0
    Specificity: float = TN / (TN + FP) if (TN + FP) != 0 else 0
    F1_score: float = 2 * (Precision * Recall) / \
        (Precision + Recall) if (Precision + Recall) != 0 else 0
    G_mean: float = math.sqrt(Recall * Specificity)
    MCC: float = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if (TP + FP) * (TP + FN) * (
        TN + FP) * (TN + FN) != 0 else 0
    PPV: float = TP / (TP + FP) if (TP + FP) != 0 else 0
    NPV: float = TN / (TN + FN) if (TN + FN) != 0 else 0
    LR_plus: float = Recall / \
        (1 - Specificity) if (1 - Specificity) != 0 else 0
    LR_minus: float = (1 - Recall) / Specificity if Specificity != 0 else 0
    Odd_ratio: float = (TP * TN) / (FP * FN) if (FP * FN) != 0 else 0
    Relative_risk: float = (TP / (TP + FP)) / (FN / (FN + TN)
                                               ) if (FN + TN) != 0 and (TP + FP) != 0 and FN != 0 else 0

    confusion_matrix_dict: dict[str, Any] = {
        "classes": [cls, "other"],
        cls: [TP, FN],
        "other": [FP, TN]
    }

    return ClassificationPredictionIndicatorsForClass(
        TP=TP,
        FP=FP,
        TN=TN,
        FN=FN,
        Recall=Recall,
        Specificity=Specificity,
        F1_score=F1_score,
        G_mean=G_mean,
        MCC=MCC,
        PPV=PPV,
        NPV=NPV,
        LR_plus=LR_plus,
        LR_minus=LR_minus,
        Odd_ratio=Odd_ratio,
        Relative_risk=Relative_risk,
        Confusion_matrix=confusion_matrix_dict
    )
