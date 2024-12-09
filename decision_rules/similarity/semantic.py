from enum import Enum

import numpy as np
import pandas as pd

from decision_rules.core.ruleset import AbstractRuleSet


class SimilarityMeasure(str, Enum):
    JACCARD = "Jaccard"
    CORRELATION = "Correlation"
    KULCZYNSKI = "Kulczynski"


def calculate_semantic_similarity_matrix(
        measure: SimilarityMeasure, ruleset1: AbstractRuleSet, ruleset2: AbstractRuleSet, dataset: pd.DataFrame
) -> np.ndarray:
    """Calculates the similarity matrix based on the specified similarity measure

    Args:
        measure (SimilarityMeasure): The similarity measure to use
        ruleset1 (AbstractRuleset): The first ruleset
        ruleset2 (AbstractRuleset): The second ruleset
        dataset (pd.DataFrame): The dataset to use for calculations

    Raises:
        ValueError: If an invalid measure is specified.

    Returns:
        np.ndarray:  The similarity matrix.
    """
    matrix1 = ruleset1.calculate_coverage_matrix(dataset).T
    matrix2 = ruleset2.calculate_coverage_matrix(dataset).T
    if measure == SimilarityMeasure.JACCARD:
        similarity_matrix = _calculate_jaccard(matrix1, matrix2)
    elif measure == SimilarityMeasure.CORRELATION:
        similarity_matrix = _calculate_corr(matrix1, matrix2)
    elif measure == SimilarityMeasure.KULCZYNSKI:
        similarity_matrix = _calculate_kulcz(matrix1, matrix2)
    else:
        raise ValueError("Invalid measure specified")
    similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
    return similarity_matrix


def _calculate_contingency_matrices(
        matrix1: np.array, matrix2: np.array
) -> tuple[np.array, np.array, np.array, np.array]:
    """
    Calculates the contingency matrices out of covered indices matrices for two rulesets.
    :param matrix1: covered indices matrix for ruleset 1
    :param matrix2: covered indices matrix for ruleset 2
    :return: contingency matrices a, b, c, d
    """
    numbers1 = matrix1.astype(np.float32)
    numbers1_inverted = np.invert(matrix1).astype(np.float32)
    numbers2 = matrix2.astype(np.float32)
    numbers2_inverted = np.invert(matrix2).astype(np.float32)
    a = np.matmul(numbers1, numbers2.T)
    b = np.matmul(numbers1, numbers2_inverted.T)
    c = np.matmul(numbers1_inverted, numbers2.T)
    d = np.matmul(numbers1_inverted, numbers2_inverted.T)
    return a, b, c, d


def _calculate_jaccard(matrix1: np.array, matrix2: np.array) -> np.array:
    a, b, c, d = _calculate_contingency_matrices(matrix1, matrix2)
    return a / (a + b + c)


def _calculate_corr(matrix1: np.array, matrix2: np.array) -> np.array:
    a, b, c, d = _calculate_contingency_matrices(matrix1, matrix2)
    return (a * d - b * c) / np.sqrt((a + b) * (a + c) * (b + d) * (c + d))


def _calculate_kulcz(matrix1: np.array, matrix2: np.array) -> np.array:
    a, b, c, d = _calculate_contingency_matrices(matrix1, matrix2)
    return 0.5 * ((a / (a + b)) + (a / (a + c)))
