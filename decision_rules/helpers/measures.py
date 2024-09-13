from __future__ import annotations

import re
from typing import Callable

from decision_rules import measures
from decision_rules.core.coverage import Coverage


def get_measure_function_by_name(measure_name: str) -> Callable[[Coverage], float]:
    """Returns function that calculates quality measure for given measure name

    Args:
        measure_name (str): Name of the quality measure in snake case or
            camel case

    Raises:
        ValueError: If measure is not supported by the package

    Returns:
        Callable[[Coverage], float]: Quality measure funciton
    """
    measure = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', measure_name)
    sanitized_measure_name = re.sub(
        '([a-z0-9])([A-Z])', r'\1_\2', measure).lower()
    # special case
    if len(sanitized_measure_name) == 2:
        sanitized_measure_name = sanitized_measure_name.replace('_', '')
    is_supported: bool = (
        hasattr(measures, sanitized_measure_name) and
        callable(getattr(measures, sanitized_measure_name))
    )
    if not is_supported:
        raise ValueError(
            f'Measure "{measure_name}" is not supported by decision rules package'
        )
    else:
        return getattr(measures, sanitized_measure_name)
