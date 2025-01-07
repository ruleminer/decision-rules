from abc import abstractmethod

import pandas as pd
import packaging.version
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.ruleset_factories._factories.abstract_factory import AbstractFactory

MINIMUM_ORANGE_VERSION = "3.35.0"


def check_if_orange_is_installed_and_correct_version():
    """
    Checks if the Orange library is installed and meets the required minimum version.
    If not, raises an ImportError with an explanation.
    """
    try:
        import Orange
    except ImportError as e:
        raise ImportError(
            "The 'ruleset_factories' extra requires 'Orange3'. "
            "Please install it with: `pip install Orange3`"
        ) from e

    actual_version = packaging.version.parse(Orange.__version__)
    required_version = packaging.version.parse(MINIMUM_ORANGE_VERSION)

    if actual_version < required_version:
        raise ImportError(
            f"The 'ruleset_factories' extra requires Orange3 version "
            f"{MINIMUM_ORANGE_VERSION} or higher. Currently installed "
            f"version is {Orange.__version__}. "
            f"Please install/update using: `pip install Orange3>={MINIMUM_ORANGE_VERSION}`"
        )

    
class AbstractOrangeCN2RuleSetFactory(AbstractFactory):
    @abstractmethod
    def make(
        self,
        model,
        X_train: pd.DataFrame,
        **kwargs,
    ) -> AbstractRuleSet:
        pass
