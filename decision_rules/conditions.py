"""
Contains logical conditions classes.
"""
from __future__ import annotations

from enum import Enum
from typing import Any
from typing import Callable
from typing import Union

import numpy as np

from decision_rules import settings
from decision_rules.core.condition import AbstractCondition


class LogicOperators(Enum):  # pylint: disable=missing-class-docstring
    CONJUNCTION = "CONJUNCTION"
    ALTERNATIVE = "ALTERNATIVE"


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
        return X[:, self.column_index].astype(str) == self.value

    def update_column_indices(self, old_to_new_attr_mapping: dict[int, int]):
        self.column_index = old_to_new_attr_mapping[self.column_index]

    def to_string(self, columns_names: list[str]) -> str:
        column_name: str = columns_names[self.column_index]
        return f'{column_name} {"!" if self.negated else ""}= {{{self.value}}}'

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, NominalCondition)
            and other.negated == self.negated
            and other.attributes == self.attributes
            and other.value == self.value
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
        left: float = float("-inf"),
        right: float = float("inf"),
        left_closed: bool = False,
        right_closed: bool = False,
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
        with np.errstate(invalid="ignore"):
            if self.left is not None:
                if not self.left_closed:
                    left_part = X[:, self.column_index] > self.left
                else:
                    left_part = X[:, self.column_index] >= self.left
            if self.right is not None:
                if not self.right_closed:
                    right_part = X[:, self.column_index] < self.right
                else:
                    right_part = X[:, self.column_index] <= self.right
            if self.left and self.right:
                return left_part & right_part
            if self.right is None:
                return left_part
            return right_part

    def update_column_indices(self, old_to_new_attr_mapping: dict[int, int]):
        self.column_index = old_to_new_attr_mapping[self.column_index]

    def to_string(self, columns_names: str) -> str:
        column_name = columns_names[self.column_index]
        if settings.CONCISE_NUMERICAL_CONDITIONS_FORM and (
            self.left == float("-inf") or self.right == float("inf")
        ):
            if self.left == float("-inf"):
                value = f"{self.right:,.{settings.FLOAT_DISPLAY_PRECISION}}"
                if self.negated:
                    sign = ">" if self.right_closed else ">="
                else:
                    sign = "<=" if self.right_closed else "<"
            else:
                value = f"{self.left:,.{settings.FLOAT_DISPLAY_PRECISION}}"
                if self.negated:
                    sign = "<" if self.left_closed else "<="
                else:
                    sign = ">=" if self.left_closed else ">"
            return f"{column_name} {sign} {value}"
        else:
            left_sign = "<" if self.left_closed else "("
            right_sign = ">" if self.right_closed else ")"
            return (
                f'{column_name} {"!" if self.negated else ""}= '
                + f"{left_sign}{self.left:,.{settings.FLOAT_DISPLAY_PRECISION}}, "
                + f"{self.right:,.{settings.FLOAT_DISPLAY_PRECISION}}{right_sign}"
            )

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, ElementaryCondition)
            and __o.attributes == self.attributes
            and __o.negated == self.negated
            and __o.left_closed == self.left_closed
            and __o.right_closed == self.right_closed
            and __o.left == self.left
            and __o.right == self.right
        )

    def __hash__(self):
        return hash(
            (
                self.negated,
                self.attributes,
                self.left,
                self.left_closed,
                self.right,
                self.right_closed,
            )
        )


class AttributesRelationCondition(AbstractCondition):
    """Condition specifying a relation between two attributes (columns).
    Possible relations are: =, !=, >, <

    Example in rule:
        IF petal_length > sepal_width THEN y = 1
    """

    class Relation(Enum):
        EQUAL = "="
        NOT_EQUAL = "!="
        GREATER = ">"
        GREATER_EQUAL = ">="
        LOWER = "<"
        LOWER_EQUAL = "<="

    def __init__(
        self,
        column_left: int,
        column_right: int,
        operator: Union[str, AttributesRelationCondition.Relation],
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

        try:
            operator = AttributesRelationCondition.Relation(operator)
        except ValueError as error:
            raise ValueError(
                "operator must be one of: "
                f"{', '.join([e.value for e in AttributesRelationCondition.Relation])}"
            ) from error
        if operator == AttributesRelationCondition.Relation.EQUAL:
            self._operator_func = lambda A, B: A == B
        elif operator == AttributesRelationCondition.Relation.NOT_EQUAL:
            self._operator_func = lambda A, B: A != B
        elif operator == AttributesRelationCondition.Relation.GREATER:
            self._operator_func = lambda A, B: A > B
        elif operator == AttributesRelationCondition.Relation.GREATER_EQUAL:
            self._operator_func = lambda A, B: A >= B
        elif operator == AttributesRelationCondition.Relation.LOWER:
            self._operator_func = lambda A, B: A < B
        elif operator == AttributesRelationCondition.Relation.LOWER_EQUAL:
            self._operator_func = lambda A, B: A <= B

    @property
    def attributes(self) -> frozenset[int]:
        return frozenset((self.column_left, self.column_right))

    def _calculate_covered_mask(self, X: np.ndarray) -> np.ndarray:
        return self._operator_func(X[:, self.column_left], X[:, self.column_right])

    def update_column_indices(self, old_to_new_attr_mapping: dict[int, int]):
        self.column_left = old_to_new_attr_mapping[self.column_left]
        self.column_right = old_to_new_attr_mapping[self.column_right]

    def to_string(self, columns_names: str) -> str:
        col_left = columns_names[self.column_left]
        col_right = columns_names[self.column_right]

        if self.negated:
            if self.operator == ">":
                operator_str = "<="
            elif self.operator == "<":
                operator_str = ">="
            elif self.operator == "=":
                operator_str = "!="
        else:
            operator_str = self.operator
        return f"{col_left} {operator_str} {col_right}"

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, AttributesRelationCondition)
            and __o.attributes == self.attributes
            and __o.column_left == self.column_left
            and __o.column_right == self.column_right
            and __o.operator == self.operator
            and __o.negated == self.negated
        )

    def __hash__(self):
        return hash(
            (
                self.negated,
                self.attributes,
                self.operator,
            )
        )


class NominalAttributesEqualityCondition(AbstractCondition):
    """Condition where nominal attributes are compared for equality

    Example:
        IF **attr1 = attr2 = attr3** THEN y = 1
    """

    def __init__(
        self,
        column_indices: list[int],
    ):
        super().__init__()
        self.column_indices: list[int] = column_indices

    @property
    def attributes(self) -> frozenset[int]:
        return frozenset(self.column_indices)

    def _calculate_covered_mask(self, X: np.ndarray) -> np.ndarray:
        arrays = X[:, list(self.column_indices)].T
        tmp = X[:, list(self.column_indices)[0]]
        mask = None
        for i in list(self.column_indices[1:]):
            if mask is None:
                mask = tmp == X[:, i]
                continue
            submask = tmp == X[:, i]
            mask = mask & submask
        return mask & np.logical_not(np.any(arrays == None, axis=0))

    def update_column_indices(self, old_to_new_attr_mapping: dict[int, int]):
        self.column_indices = [old_to_new_attr_mapping[i]
                               for i in self.column_indices]

    def to_string(self, columns_names: list[str]) -> str:
        operator: str = " != " if self.negated else " = "
        return operator.join(columns_names[index] for index in self.column_indices)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, self.__class__):
            return False
        return __o.column_indices == self.column_indices and __o.negated == self.negated

    def __hash__(self):
        return hash(
            (
                self.negated,
                self.attributes,
            )
        )


class DiscreteSetCondition(AbstractCondition):
    """Condition checking if the value of an nominal attribute belongs to a given set
    of values.

    Example:
        IF **color = {red, green}** THEN y = 1
    """

    def __init__(
        self,
        column_index: int,
        values_set: set[str],
    ):
        super().__init__()

        self.column_index: int = column_index
        self.values_set: set[str] = values_set

    @property
    def attributes(self) -> frozenset[int]:
        return frozenset({self.column_index})

    def _calculate_covered_mask(self, X: np.ndarray) -> np.ndarray:
        return np.any([X[:, self.column_index] == e for e in self.values_set], axis=0)

    def update_column_indices(self, old_to_new_attr_mapping: dict[int, int]):
        self.column_index = old_to_new_attr_mapping[self.column_index]

    def to_string(self, columns_names: list[str]) -> str:
        column_name: str = columns_names[self.column_index]
        return column_name + " = {" + ", ".join(self.values_set) + "}"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, self.__class__):
            return False
        return (
            __o.column_index == self.column_index
            and __o.values_set == self.values_set
            and __o.negated == self.negated
        )

    def __hash__(self):
        return hash((self.negated, self.attributes, frozenset(self.values_set)))


class CompoundCondition(AbstractCondition):
    """Condition specifying logical conjunction or alternative
    of other conditions.

    Example in rule:
        IF petal_length = <-0.015, 0.185> AND sepal_width = <-0.047, 0.253> THEN y = 1
    """

    def __init__(
        self,
        subconditions: list[ElementaryCondition],
        logic_operator: LogicOperators = LogicOperators.CONJUNCTION,
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
        return frozenset().union(
            *[subcondition.attributes for subcondition in self.subconditions]
        )

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

    def update_column_indices(self, old_to_new_attr_mapping: dict[int, int]):
        for condition in self.subconditions:
            condition.update_column_indices(old_to_new_attr_mapping)

    def to_string(self, columns_names: list[str]) -> str:
        tmp = []
        for subcondition in self.subconditions:
            subconditions_str: str = f"{subcondition.to_string(columns_names)}"
            if isinstance(subcondition, CompoundCondition):
                subconditions_str = f"({subconditions_str})"
            tmp.append(subconditions_str)
        operator_str = (
            " AND " if self.logic_operator == LogicOperators.CONJUNCTION else " OR "
        )
        tmp = operator_str.join(tmp)
        if self.negated:
            negation_str = "!" if self.negated else ""
            return f"{negation_str}({tmp})"
        return tmp

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CompoundCondition)
            and other.logic_operator == self.logic_operator
            and frozenset(other.subconditions) == frozenset(self.subconditions)
        )

    def __hash__(self):
        return hash(
            (
                self.negated,
                self.attributes,
                self.logic_operator,
                hash((s.__hash__() for s in self.subconditions)),
            )
        )
