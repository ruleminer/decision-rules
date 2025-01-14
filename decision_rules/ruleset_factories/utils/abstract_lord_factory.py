from abc import abstractmethod

import pandas as pd
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.ruleset_factories._factories.abstract_factory import AbstractFactory


class AbstractLordRuleSetFactory(AbstractFactory):
    """
    Abstract class for LORD rule set factories.

    Methods:
    --------
    make(
        model: list[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs,
    ) -> AbstractRuleSet:
        Abstract method to create a `decision-rules` rule set.
        Implementations for all types of problems parse a list of rules
        from LORD to text format recognized by `TextRuleSetFactory`
        and pass it to corresponding text factory for a given problem type.
    """
    @abstractmethod
    def make(
        self,
        model: list[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs,
    ) -> AbstractRuleSet:
        pass
