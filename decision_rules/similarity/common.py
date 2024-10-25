import numpy as np
import pandas as pd

from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.parsers import FilterList
from decision_rules.parsers import FilterToMaskProcessor
from decision_rules.parsers import RuleToFilterParser


def _get_covered_index_matrix(dataset: pd.DataFrame, ruleset: AbstractRuleSet) -> np.array:
    """
    Calculates a matrix of shape (n_examples, n_rules) where each row represents an example
    and each column represents a rule.
    :param dataset: dataframe with the dataset
    :param ruleset: ruleset to calculate the matrix for
    :return: matrix of shape (n_examples, n_rules) where each row represents an example
    """
    filters: FilterList = RuleToFilterParser(
        ruleset).parse_ruleset_to_filters()
    processor = FilterToMaskProcessor()
    masks = np.array([
        processor.process(dataset, filters_).to_numpy()
        for filters_ in filters.filters
    ])
    return masks
