"""
Contains AttributeImportance class for determining importances of condtions in RuleSet.
"""
from __future__ import annotations

import re
from abc import ABC
from collections import defaultdict


class AbstractRuleSetAttributeImportances(ABC):
    """Abstract AtrributeImportance allowing to determine importances of atrribute in RuleSet
    """

    def calculate_importances_base_on_conditions(self, conditions_importances: list[dict]) -> dict[str, float]:
        """Calculate importances of attributes based on condition importances in RuleSet
        """
        attributes_importances = defaultdict(float)

        for condition_importance in conditions_importances:
            attribute_names = condition_importance['attributes']
            value = condition_importance['importance']
            for attribute_name in attribute_names:
                attributes_importances[attribute_name] += value

        attributes_importances = dict(
            sorted(attributes_importances.items(), key=lambda item: item[1], reverse=True))

        return attributes_importances
