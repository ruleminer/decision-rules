"""
Contains helpers classes.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Union

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

        TOP_LEVEL: str = "top_level"
        SPLIT: str = "split"
        NESTED: str = "nested"

    def __init__(self, conditions: Iterable[AbstractCondition]) -> None:
        """
        Args:
            conditions (list[AbstractCondition]): conditions
        """
        self.conditions: Iterable[AbstractCondition] = conditions

    def _prepare_conditions_set(
        self, method: Union[str, ConditionalDatasetTransformer.Methods]
    ) -> set[AbstractCondition]:
        conditions: set[AbstractCondition] = set()
        try:
            method = ConditionalDatasetTransformer.Methods(method)
        except ValueError as e:
            raise ValueError(
                '"method" parameter should have one of the following value: ['
                + ", ".join(
                    f'"{e.value}"' for e in ConditionalDatasetTransformer.Methods
                )
                + f'] but value: "{method}" was passed.'
            ) from e
        if method == ConditionalDatasetTransformer.Methods.TOP_LEVEL:
            conditions = self.conditions
        elif method == ConditionalDatasetTransformer.Methods.SPLIT:
            for condition in self.conditions:
                conditions.add(condition)
                conditions.update(condition.subconditions)
        elif method == ConditionalDatasetTransformer.Methods.NESTED:

            def _add_condition(
                conditions: set[AbstractCondition], condition: AbstractCondition
            ):
                conditions.add(condition)
                for subcondition in condition.subconditions:
                    _add_condition(conditions, subcondition)

            for condition in self.conditions:
                _add_condition(conditions, condition)
        return conditions

    def transform(
        self,
        X: np.ndarray,
        column_names: Iterable[str],
        method: ConditionalDatasetTransformer.Methods = "top_level",
    ) -> pd.DataFrame:
        """Transform dataset with set of conditions producing binary dataset.

        Args:
            X (np.ndarray): X
            column_names (list[str]): names of columns
            method (ConditionalDatasetTransformer.Methods): controls how to generate columns.
                `top_level`: passed conditions as columns,
                `split`: passed conditions and their subconditions as columns,
                `nested`: all passed conditions and their subconditions recursivly.

                Defaults is `top_level`.

        Returns:
            pd.DataFrame: transformed binary dataset
        """
        conditions: set[AbstractCondition] = self._prepare_conditions_set(method)
        new_columns_names: list[str] = [
            condition.to_string(column_names) for condition in conditions
        ]
        X_t = np.zeros((X.shape[0], len(conditions)))  # pylint: disable=invalid-name
        for i, condition in enumerate(conditions):
            X_t[:, i] = condition.covered_mask(X)

        df = pd.DataFrame(X_t, columns=new_columns_names)
        return df.astype("uint")
