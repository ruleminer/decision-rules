"""Module containing helper functions for tests.
"""
import math

import numpy as np


def check_if_any_of_dict_value_is_nan(dictionary: dict) -> bool:
    """Checks if any value in dictionary is NaN

    Args:
        dictionary (dict): dictionary to check

    Returns:
        bool: true if any value in dictionary is NaN
    """
    was_nan: bool = False
    for key, value in dictionary.items():
        if math.isnan(value) or value is None:
            print(f'"{key}" = {value}')
            was_nan = True
    return was_nan


def compare_survival_prediction(
    actual: np.ndarray,
    expected: np.ndarray
) -> bool:
    for actual, expected in zip(actual, expected):
        times_the_same = len(actual['times']) == len(expected['times']) and np.allclose(
            actual['times'], expected['times'], atol=1.0e-10)
        probs_the_same = len(actual['probabilities']) == len(expected['probabilities']) and np.allclose(
            actual['probabilities'], expected['probabilities'], atol=1.0e-10)
        if not times_the_same or not probs_the_same:
            return False
    return True
