"""
Contains AttributeImportance class for determining importances of condtions in RuleSet.
"""
from __future__ import annotations

from decision_rules.importances._core import AbstractRuleSetAttributeImportances


class RegressionRuleSetAttributeImportances(AbstractRuleSetAttributeImportances):
    """Regression AtrributeImportance allowing to determine importances of atrribute in RegressionRuleSet
    """
