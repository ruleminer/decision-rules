from abc import ABC
from abc import abstractmethod
from typing import Any

import pandas as pd
from decision_rules.core.ruleset import AbstractRuleSet


class AbstractFactory(ABC):
    """Base interface for RuleSet model factory
    """

    @abstractmethod
    def make(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> AbstractRuleSet:
        pass
