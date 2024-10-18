"""
Contains base abstract condition class
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager

import numpy as np
import pandas as pd


class AbstractCondition(ABC):
    """Abstract class for logical conditions specifying their public interface.
    Every conditions class should extend this class.
    """

    def __init__(self) -> None:
        self.negated: bool = False
        self.subconditions: list[AbstractCondition] = []

        self.__cached_covered_mask: np.ndarray = None
        self.__cached_uncovered_mask: np.ndarray = None
        self.cached: bool = False

    @contextmanager
    def cache(self, recursive: bool = False):
        """Caches condition covered and uncovered examples masks to
        prevent their recalculation. It automatically enable cache for
        all condition's subconditions.

        Examples
        --------
        >>> with condition.cache()
        >>>     covered_mask = condition.covered_mask(X)
        >>>     ...

        Yields:
            None: none
        """
        was_already_cached: bool = self.cached
        if self.cached:
            yield None
        else:
            self.cached = True
            if recursive:
                for subcondition in self.subconditions:
                    subcondition.cached = True  # pylint: disable=protected-access
            try:
                yield None
            finally:
                if not was_already_cached:
                    if recursive:
                        for subcondition in self.subconditions:
                            subcondition.invalidate_cache()
                    self.invalidate_cache()

    def invalidate_cache(self):
        """Disable covered and uncovered examples masks cache.
        """
        self.cached = False
        self.__cached_covered_mask = None
        self.__cached_uncovered_mask = None
        for subcondition in self.subconditions:
            subcondition.invalidate_cache()

    @property
    @abstractmethod
    def attributes(self) -> frozenset[int]:
        """
        Returns:
            frozenset[int]: condition attributes
        """

    @abstractmethod
    def __eq__(self, __o: object) -> bool:
        pass

    @abstractmethod
    def __hash__(self):
        """
        Notes
        -----
        Conditions ARE NOT hashable in common sense of this word and
        therefore should NOT be used as dictionary keys or sets memebers.
        This method is only used for quicker comparation of rules and rulesets.
        """

    @abstractmethod
    def _calculate_covered_mask(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X (np.ndarray)
        Returns:
            np.ndarray: 1 dimensional numpy array of booleans specifying
                whether given examples is covered by a condition or not.
        """

    def _calculate_uncovered_mask(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X (np.ndarray)
        Returns:
            np.ndarray: 1 dimensional numpy array of booleans specifying
                whether given examples is covered by a condition or not.
        """
        valid_examples_mask = np.all(
            pd.notnull(X[:, list(self.attributes)]),
            axis=1
        )
        return np.logical_not(self._calculate_covered_mask(X)) & valid_examples_mask

    def covered_mask(self, X: np.ndarray) -> np.ndarray:
        """Calculates covered examples mask

        Args:
            X (np.ndarray)
        Returns:
            np.ndarray: 1 dimensional numpy array of booleans specifying
                whether given examples is covered by a condition or not.
        """
        if self.cached and self.__cached_covered_mask is not None:
            return self.__cached_covered_mask
        if self.cached and self.__cached_uncovered_mask is not None:
            return np.logical_not(self.__cached_uncovered_mask)
        if self.negated:
            covered_mask = self._calculate_uncovered_mask(X)
        else:
            covered_mask = self._calculate_covered_mask(X)
        if self.cached:
            self.__cached_covered_mask = covered_mask
        return covered_mask

    def uncovered_mask(self, X: np.ndarray) -> np.ndarray:
        """Calculates uncovered examples mask - negation of covered mask

        See Also
        --------
        conditions.conditions.AbstractCondition.covered_mask

        Args:
            X (np.ndarray)
        Returns:
            np.ndarray: 1 dimensional numpy array of booleans specifying
                whether given examples is uncovered by a condition or not.
        """
        if self.cached and self.__cached_uncovered_mask is not None:
            return self.__cached_uncovered_mask
        if self.cached and self.__cached_covered_mask is not None:
            return np.logical_not(self.__cached_covered_mask)
        if self.negated:
            uncovered_mask = self._calculate_covered_mask(X)
        else:
            uncovered_mask = self._calculate_uncovered_mask(X)
        if self.cached:
            self.__cached_uncovered_mask = uncovered_mask
        return uncovered_mask

    @abstractmethod
    def to_string(self, columns_names: list[str]) -> str:
        """
        Args:
            columns_names (list[str])

        Returns:
            str: condition string representation
        """
