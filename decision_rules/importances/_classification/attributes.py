"""
Contains AttributeImportance class for determining importances of condtions in RuleSet.
"""
from __future__ import annotations

import re
from collections import defaultdict

from decision_rules.importances._core import AbstractRuleSetAttributeImportances


class ClassificationRuleSetAttributeImportances(AbstractRuleSetAttributeImportances):
    """Classiciation AtrributeImportance allowing to determine importances of atrribute in ClassificationRuleSet
    """

    def calculate_importances_base_on_conditions(self, condition_importances_by_class: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
        """Calculate importances of attributes based on condition importances in RuleSet for each class."""
        attributes_importances_by_class = {}

        for class_name, conditions_importances_list in condition_importances_by_class.items():
            attributes_importances_for_class = defaultdict(float)

            for condition_importance_dict in conditions_importances_list:
                attribute_names = condition_importance_dict['attributes']
                value = condition_importance_dict['importance']
                for attribute_name in attribute_names:
                    attributes_importances_for_class[attribute_name] += value

            attributes_importances_by_class[class_name] = dict(sorted(
                attributes_importances_for_class.items(), key=lambda item: item[1], reverse=True))

        return attributes_importances_by_class
