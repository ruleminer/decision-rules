import numpy as np
import pandas as pd
from decision_rules.core.ruleset import AbstractRuleSet
from rolap_data_storage.abstract.reader import FilterList
from rolap_data_storage.implementations.aws.reader import FilterToMaskProcessor
from rolap_data_storage.parsers.ruleset import RuleToFilterParser


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
