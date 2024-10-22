import numpy as np


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
