"""
Contains logical conditions classes.
"""
from __future__ import annotations

from enum import Enum
from typing import Any
from typing import Callable

import numpy as np
from decision_rules import settings
from decision_rules.core.condition import AbstractCondition


class LogicOperators(Enum):  # pylint: disable=missing-class-docstring
    CONJUNCTION = 'CONJUNCTION'
    ALTERNATIVE = 'ALTERNATIVE'


class NominalCondition(AbstractCondition):
    """Class for elementary condition for nominal attributes.
    It took following form:
        IF {ATTRIBUTE} = {VALUE} THEN y = {DECISION}
    Example:
        IF gender = 1 THEN y = 1
    """

    def __init__(
        self,
        column_index: int,
        value: str,
    ) -> None:
        super().__init__()

        self.value: str = str(value)
        self.column_index: int = column_index

    @property
    def attributes(self) -> frozenset[int]:
        return frozenset((self.column_index,))

    def _calculate_covered_mask(self, X: np.ndarray) -> np.ndarray:
        return (X[:, self.column_index].astype(str) == self.value)

    def to_string(self, columns_names: list[str]) -> str:
        column_name: str = columns_names[self.column_index]
        return f'{column_name} {"!" if self.negated else ""}= {{{self.value}}}'

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, NominalCondition) and
            other.negated == self.negated and
            other.attributes == self.attributes and
            other.value == self.value
        )

    def __hash__(self):
        return hash((self.negated, self.attributes, self.value))


class ElementaryCondition(AbstractCondition):
    """Numerical attributes condition.

    It has left and right boundary and takes following form:
        {LEFT} < {ATTRIBUTE} < {RIGHT}

    Example in rule:
        IF petal_length = <-0.015, 0.185> THEN y = 1
    """

    def __init__(
        self,
        column_index: int,
        left: float = float('-inf'),
        right: float = float('inf'),
        left_closed: bool = False,
        right_closed: bool = False
    ) -> None:
        """
        Args:
            column_index (int): condition attribute column index in dataset
            left (float, optional): left boundary of the interval. Defaults to float('-inf').
            right (float, optional): right boundary of the interval. Defaults to float('inf').
            left_closed (bool, optional): whether the interval is closed on the left.
                Defaults to False.
            right_closed (bool, optional): whether the interval is closed on the right.
                Defaults to False.
        """
        super().__init__()

        self.left: float = left
        self.right: float = right
        self.left_closed: bool = left_closed
        self.right_closed: bool = right_closed
        self.column_index: int = column_index

    @property
    def attributes(self) -> frozenset[int]:
        return frozenset((self.column_index,))

    def _calculate_covered_mask(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(invalid='ignore'):
            if self.left is not None:
                if not self.left_closed:
                    left_part = (X[:, self.column_index] > self.left)
                else:
                    left_part = (X[:, self.column_index] >= self.left)
            if self.right is not None:
                if not self.right_closed:
                    right_part = (X[:, self.column_index] < self.right)
                else:
                    right_part = (X[:, self.column_index] <= self.right)
            if self.left and self.right:
                return left_part & right_part
            if self.right is None:
                return left_part
            return right_part

    def to_string(self, columns_names: str) -> str:
        column_name = columns_names[self.column_index]
        if settings.CONCISE_NUMERICAL_CONDITIONS_FORM and (
            self.left == float('-inf') or self.right == float('inf')
        ):
            if self.left == float('-inf'):
                value = f'{self.right:,.{settings.FLOAT_DISPLAY_PRECISION}}'
                if self.negated:
                    sign = '>' if self.right_closed else '>='
                else:
                    sign = '<=' if self.right_closed else '<'
            else:
                value = f'{self.left:,.{settings.FLOAT_DISPLAY_PRECISION}}'
                if self.negated:
                    sign = '<' if self.left_closed else '<='
                else:
                    sign = '>=' if self.left_closed else '>'
            return f'{column_name} {sign} {value}'
        else:
            left_sign = '<' if self.left_closed else '('
            right_sign = '>' if self.right_closed else ')'
            return (
                f'{column_name} {"!" if self.negated else ""}= ' +
                f'{left_sign}{self.left:,.{settings.FLOAT_DISPLAY_PRECISION}}, ' +
                f'{self.right:,.{settings.FLOAT_DISPLAY_PRECISION}}{right_sign}'
            )

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, ElementaryCondition) and
            __o.attributes == self.attributes and
            __o.negated == self.negated and
            __o.left_closed == self.left_closed and
            __o.right_closed == self.right_closed and
            __o.left == self.left and
            __o.right == self.right
        )

    def __hash__(self):
        return hash((
            self.negated,
            self.attributes,
            self.left,
            self.left_closed,
            self.right,
            self.right_closed
        ))


class AttributesCondition(AbstractCondition):
    """Condition specifying a relation between two attributes (columns).
    Possible relations are: =, !=, >, <

    Example in rule:
        IF petal_length > sepal_width THEN y = 1
    """

    def __init__(
        self,
        column_left: int,
        column_right: int,
        operator: str,
    ) -> None:
        """
        Args:
            column_left (int): left-hand side column
            column_right (int): right-hand side column
            operator (str): operator specyfing relation between two columns
        """
        super().__init__()

        self.column_left: int = column_left
        self.column_right: int = column_right
        self.operator: str = operator
        self._operator_func: Callable[[Any, Any], np.ndarray] = None

        if operator == '<':
            self._operator_func = lambda A, B: A < B
        if operator == '>':
            self._operator_func = lambda A, B: A > B
        if operator == '=':
            self._operator_func = lambda A, B: A == B

    @property
    def attributes(self) -> frozenset[int]:
        return frozenset((self.column_left, self.column_right))

    def _calculate_covered_mask(self, X: np.ndarray) -> np.ndarray:
        return self._operator_func(X[:, self.column_left], X[:, self.column_right])

    def to_string(self, columns_names: str) -> str:
        col_left = columns_names[self.column_left]
        col_right = columns_names[self.column_right]

        if self.negated:
            if self.operator == '>':
                operator_str = '<='
            elif self.operator == '<':
                operator_str = '>='
            elif self.operator == '=':
                operator_str = '!='
        else:
            operator_str = self.operator
        return f'{col_left} {operator_str} {col_right}'

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, AttributesCondition) and
            __o.attributes == self.attributes and
            __o.column_left == self.column_left and
            __o.column_right == self.column_right and
            __o.operator == self.operator and
            __o.negated == self.negated
        )

    def __hash__(self):
        return hash((
            self.negated,
            self.attributes,
            self.operator,
        ))


class CompoundCondition(AbstractCondition):
    """Condition specyfing logical conjunction or alternative
    of other conditions.

    Example in rule:
        IF petal_length = <-0.015, 0.185> AND sepal_width = <-0.047, 0.253> THEN y = 1
    """

    def __init__(
        self,
        subconditions: list[ElementaryCondition],
        logic_operator: LogicOperators = LogicOperators.CONJUNCTION
    ) -> None:
        """
        Args:
            subconditions (list[ElementaryCondition])
            logic_operator (LogicOperators, optional): Defaults to LogicOperators.CONJUNCTION.
        """
        super().__init__()

        self.subconditions: list[ElementaryCondition] = subconditions
        self.logic_operator: LogicOperators = logic_operator

    @property
    def attributes(self) -> frozenset[int]:
        return frozenset().union(*[
            subcondition.attributes for subcondition in self.subconditions
        ])

    def _calculate_covered_mask(self, X: np.ndarray) -> np.ndarray:
        if len(self.subconditions) == 0:
            return np.ones(X.shape[0], dtype=bool)
        covered_mask = self.subconditions[0].covered_mask(X)
        if self.logic_operator == LogicOperators.CONJUNCTION:
            for i in range(1, len(self.subconditions)):
                covered_mask &= self.subconditions[i].covered_mask(X)
        else:
            for i in range(1, len(self.subconditions)):
                covered_mask |= self.subconditions[i].covered_mask(X)
        return covered_mask

    def to_string(self, columns_names: list[str]) -> str:
        tmp = []
        for subcondition in self.subconditions:
            subconditions_str: str = f'{subcondition.to_string(columns_names)}'
            if isinstance(subcondition, CompoundCondition):
                subconditions_str = f'({subconditions_str})'
            tmp.append(subconditions_str)
        operator_str = ' AND ' if self.logic_operator == LogicOperators.CONJUNCTION else ' OR '
        tmp = operator_str.join(tmp)
        if self.negated:
            negation_str = '!' if self.negated else ''
            return f'{negation_str}({tmp})'
        return tmp

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CompoundCondition) and
            other.logic_operator == self.logic_operator and
            frozenset(other.subconditions) == frozenset(self.subconditions)
        )

    def __hash__(self):
        return hash((
            self.negated,
            self.attributes,
            self.logic_operator,
            hash((s.__hash__() for s in self.subconditions))
        ))
