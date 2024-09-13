from enum import Enum

import numpy as np
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.similarity.semantic import calculate_semantic_similarity_matrix
from decision_rules.similarity.semantic import SimilarityMeasure
from decision_rules.similarity.syntactic import SyntacticRuleSimilarityCalculator


class SimilarityType(str, Enum):
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"


def calculate_rule_similarity(
        ruleset1: AbstractRuleSet, ruleset2: AbstractRuleSet, dataset,
        similarity_type: SimilarityType, measure: SimilarityMeasure = None
) -> np.ndarray:
    """
    Calculate the rule similarity between two sets of rules based on the specified measure and type.

    :param ruleset1: AbstractRuleSet
    :param ruleset2: AbstractRuleSet
    :param dataset: pd.DataFrame
    :param similarity_type: SimilarityType - semantic or syntactic
    :param measure: for semantic similarity

    :return: array (matrix) of similarities
    """
    if similarity_type == SimilarityType.SEMANTIC:
        if measure is None:
            raise ValueError(
                "Measure must be specified for semantic similarity")
        return calculate_semantic_similarity_matrix(measure, ruleset1, ruleset2, dataset)
    elif similarity_type == SimilarityType.SYNTACTIC:
        calculator = SyntacticRuleSimilarityCalculator(
            ruleset1, ruleset2, dataset)
        return calculator.calculate()
    else:
        raise ValueError("Invalid similarity type specified")
