# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

import numpy as np
from decision_rules.conditions import AbstractCondition


class TestingCondition(AbstractCondition):

    def __init__(self) -> None:
        super().__init__()
        self.colum_index = 0
        self.code: int = 0

    @property
    def attributes(self) -> frozenset[int]:
        return {0}

    def _calculate_covered_mask(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.colum_index] != 0

    def to_string(self, columns_names: list[str]) -> str:
        return ''

    def __hash__(self):
        return hash((self.colum_index, self.code))

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, TestingCondition) and __o.code == self.code


class TestAbstractCondition(unittest.TestCase):

    def test_cache(self):
        X: np.ndarray = np.array([1, 0, 1, 1, 0, 1, 0])
        X = np.expand_dims(X, axis=1)
        condition = TestingCondition()  # pylint: disable=abstract-class-instantiated

        expected_covered_mask = X.astype(bool)[:, 0]
        real_covered_mask = condition.covered_mask(X)
        self.assertTrue(
            np.array_equal(real_covered_mask, expected_covered_mask),
            'Covered mask should work the same as _calculate_covered_mask without cache'
        )

        with condition.cache():
            old_covered_mask: np.ndarray = condition.covered_mask(X)
            X[0] = 0
            real_covered_mask = condition.covered_mask(X)
            self.assertTrue(
                np.array_equal(real_covered_mask, old_covered_mask),
                '"covered_mask" should return last cached mask which may not always be up valid'
            )

        expected_covered_mask = X.astype(bool)[:, 0]
        real_covered_mask = condition.covered_mask(X)
        self.assertTrue(
            np.array_equal(real_covered_mask, expected_covered_mask),
            'After invalidating cache mask should be recalculated on new data'
        )


if __name__ == '__main__':
    unittest.main()
