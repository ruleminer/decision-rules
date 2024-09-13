"""
Contains classification rule and conclusion classes.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from decision_rules.core.condition import AbstractCondition
from decision_rules.core.rule import AbstractConclusion
from decision_rules.core.rule import AbstractRule


class ClassificationConclusion(AbstractConclusion):
    """Conclusion part of the classification rule

    Args:
        AbstractConclusion (_type_):
    """

    def __init__(
        self,
        value: Union[str, int],
        column_name: str
    ) -> None:
        super().__init__(value, column_name)

    def positives_mask(self, y: np.ndarray) -> np.ndarray:
        return y == self.value

    @staticmethod
    def make_empty(column_name: str) -> ClassificationConclusion:  # pylint: disable=invalid-name
        return ClassificationConclusion(column_name=column_name, value='')

    def is_empty(self) -> bool:
        return self.value == ''

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AbstractConclusion) and
            other.column_name == self.column_name and
            other.value == self.value
        )

    def __hash__(self) -> int:
        return hash((self.column_name, self.value))

    def __str__(self) -> str:
        return f'{self.column_name} = {self.value}'


class ClassificationRule(AbstractRule):
    """Classification decision rule.
    """

    def __init__(
        self,
        premise: AbstractCondition,
        conclusion: ClassificationConclusion,
        column_names: list[str]
    ) -> None:
        self.conclusion: ClassificationConclusion = conclusion
        super().__init__(premise, conclusion, column_names)
