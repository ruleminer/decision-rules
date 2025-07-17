from decision_rules.core.ruleset import AbstractRuleSet

def get_condition_frequent(ruleset: AbstractRuleSet) -> dict[str, int]:
    """
    Returns a dictionary with the string representation of each condition and its occurrence count across all rules in the ruleset.
    """
    condition_counts: dict[str, int] = {}

    def traverse(condition: object) -> None:
        if hasattr(condition, "subconditions") and condition.subconditions:
            for subcondition in condition.subconditions:
                traverse(subcondition)
        else: 
            key = condition.to_string(ruleset.column_names)
            condition_counts[key] = condition_counts.get(key, 0) + 1

    for rule in ruleset.rules:
        traverse(rule.premise)
    return condition_counts

def get_attribute_frequent(ruleset: AbstractRuleSet) -> dict[str, int]:
    """
    Returns a dictionary with attribute names and the number of times they appear in all conditions
    of the rules in the ruleset. 
    """
    attribute_counts: dict[str, int] = {}

    def traverse(condition: object) -> None:
        # If the condition is compound, traverse its subconditions.
        if hasattr(condition, "subconditions") and condition.subconditions:
            for subcondition in condition.subconditions:
                traverse(subcondition)
        else:
            # Check for AttributesCondition which uses column_left and column_right
            if hasattr(condition, "column_left") and hasattr(condition, "column_right"):
                if ruleset.column_names is not None:
                    attr_left = ruleset.column_names[condition.column_left]
                    attr_right = ruleset.column_names[condition.column_right]
                else:
                    attr_left = str(condition.column_left)
                    attr_right = str(condition.column_right)
                attribute_counts[attr_left] = attribute_counts.get(attr_left, 0) + 1
                attribute_counts[attr_right] = attribute_counts.get(attr_right, 0) + 1
            # Otherwise, if condition has a single column_index attribute
            elif hasattr(condition, "column_index"):
                if ruleset.column_names is not None:
                    attr_name = ruleset.column_names[condition.column_index]
                else:
                    attr_name = str(condition.column_index)
                attribute_counts[attr_name] = attribute_counts.get(attr_name, 0) + 1

    for rule in ruleset.rules:
        traverse(rule.premise)
    return attribute_counts
