"""
Contains helpers classes.
"""
from __future__ import annotations

from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd
from decision_rules.conditions import AbstractCondition


class ConditionalDatasetTransformer:
    """Helper class transforming dataset with given set of conditions. It produces
    binary dataset showing conditions coverage.
    """

    class Methods(Enum):
        """Methods of how to extract conditions from rules.

        Args:
            Enum (_type_): _description_
        """
        TOP_LEVEL: str = 'top_level'
        SPLIT: str = 'split'
        NESTED: str = 'nested'

    def __init__(self, conditions: Iterable[AbstractCondition]) -> None:
        """
        Args:
            conditions (list[BaseCondition]): conditions
        """
        self.conditions: Iterable[AbstractCondition] = conditions

    def _prepare_conditions_set(
        self,
        method: ConditionalDatasetTransformer.Methods
    ) -> set[AbstractCondition]:
        conditions: set[AbstractCondition] = set()
        if method == ConditionalDatasetTransformer.Methods.TOP_LEVEL:
            conditions = self.conditions
        elif method == ConditionalDatasetTransformer.Methods.SPLIT:
            for condition in self.conditions:
                conditions.add(conditions)
                for subcondition in condition.subconditions:
                    conditions.add(subcondition)
        elif method == ConditionalDatasetTransformer.Methods.NESTED:
            def _add_condition(
                    conditions: set[AbstractCondition],
                    condition: AbstractCondition
            ):
                conditions.add(condition)
                for subcondition in condition.subconditions:
                    _add_condition(conditions, subcondition)
            for condition in self.conditions:
                _add_condition(conditions, condition)
        else:
            raise ValueError(
                '"method" parameter should have one of the following value: [' +
                ', '.join(
                    f'"{e.value}"' for e in ConditionalDatasetTransformer.Methods
                ) +
                f'] but value: "{method}" was passed.'
            )
        return conditions

    def transform(
            self,
            X: np.ndarray,
            column_names: Iterable[str],
            method: ConditionalDatasetTransformer.Methods = 'top_level'
    ) -> pd.DataFrame:
        """Transform dataset with set of conditions producing binary dataset.

        Args:
            X (np.ndarray): X
            column_names (list[str]): names of columns
            method (ConditionalDatasetTransformer.Methods): controls how to generate colums.
                "top_level": passed conditions as columns,
                "split": passed conditions and their subconditions as columns,
                "nested": all passed conditions and their subconditions recursivly

        Returns:
            pd.DataFrame: transformed binary dataset
        """
        new_columns_names: list[str] = [
            condition.to_string(column_names) for condition in self.conditions
        ]
        conditions: set[AbstractCondition] = self._prepare_conditions_set(
            method)
        X_t = np.zeros(  # pylint: disable=invalid-name
            (X.shape[0], len(conditions))
        )
        for i, condition in enumerate(conditions):
            X_t = condition.evaluate(  # pylint: disable=invalid-name
                X_t, X, column_index=i
            )

        df = pd.DataFrame(X_t, columns=new_columns_names)
        return df.astype(int)
