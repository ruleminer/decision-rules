import numpy as np
import pandas as pd
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.similarity.common import _get_covered_index_matrix


def calculate_ruleset_similarity(
        ruleset1: AbstractRuleSet, ruleset2: AbstractRuleSet, dataset: pd.DataFrame
) -> float:
    """
    Calculates the similarity between two rulesets based on the number of pairs of examples
    that are covered by the same rules in both rulesets.

    :param ruleset1: first ruleset
    :param ruleset2: second ruleset
    :param dataset: dataset to calculate the similarity for
    :return: similarity between the rulesets
    """
    # calculate covered indices matrices for both rulesets
    matrix1 = _get_covered_index_matrix(dataset, ruleset1)
    matrix2 = _get_covered_index_matrix(dataset, ruleset2)
    # Make sure they have the same number of examples
    if matrix1.shape[1] != matrix2.shape[1]:
        raise ValueError("Wrong dimensions of the matrices")
    # calculate the number of possible pairs in the dataset overall (denominator)
    dataset_size = matrix1.shape[1]
    all_dataset_pairs = dataset_size * (dataset_size - 1) / 2
    # calculate matrices of pairs - whether they are covered by at least one common rule
    a1 = np.matmul(matrix1.T, matrix1)
    a2 = np.matmul(matrix2.T, matrix2)
    # complementary - whether a pair is covered by no common rule
    b1 = np.logical_not(a1)
    b2 = np.logical_not(a2)
    # find overlap of those pairs between rulesets
    a = np.logical_and(a1, a2)
    b = np.logical_and(b1, b2)
    # calculate numbers of pairs a and b
    a = np.tril(a).sum() - a.trace()
    b = np.tril(b).sum() - b.trace()
    # return result
    return (a + b) / all_dataset_pairs
