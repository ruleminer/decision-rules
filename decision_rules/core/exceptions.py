class InvalidStateError(Exception):
    """Error indicating that object in in a wrong state to perform certain
    operation.

    Args:
        Exception (_type_): _description_
    """
class RulesetFactoriesException(Exception):
    """
    Base exception for all errors raised from ruleset factories.
    """
    pass


class RuleConclusionFormatException(RulesetFactoriesException):
    def __init__(self, conclusion_part: str):
        self.detail = {"conclusion_part": conclusion_part}
        message = f"Rule conclusion format is incorrect: {conclusion_part}"
        super().__init__(message)


class RuleConclusionFloatConversionException(RulesetFactoriesException):
    def __init__(self):
        message = f"Error converting rule conclusion values to float"
        super().__init__(message)


class DecisionAttributeMismatchException(RulesetFactoriesException):
    def __init__(self, given_attribute: str, expected_attribute: str):
        self.detail = {
            "given_attribute": given_attribute,
            "expected_attribute": expected_attribute
        }
        message = (
            f"Decision attribute '{given_attribute}' does not match the expected attribute '{expected_attribute}'"
        )
        super().__init__(message)


class InvalidMeasureNameException(RulesetFactoriesException):
    def __init__(self):
        message = "measure_name must be either a string or a function"
        super().__init__(message)

class InvalidSurvivalTimeAttributeException(RulesetFactoriesException):
    def __init__(self, attribute: str):
        self.detail = {"attribute": attribute}
        message = f"Invalid survival time attribute name: {attribute}"
        super().__init__(message)

class InvalidConditionFormatException(RulesetFactoriesException):
    def __init__(self, condition_str: str):
        self.detail = {"condition_str": condition_str}
        message = f"Invalid condition format: {condition_str}"
        super().__init__(message)

class AttributeNotFoundException(RulesetFactoriesException):
    def __init__(self, attribute_name: str):
        self.detail = {"attribute_name": attribute_name}
        message = f"Attribute '{attribute_name}' not found in dataset columns"
        super().__init__(message)

class InvalidNumericValueException(RulesetFactoriesException):
    def __init__(self, operator: str, value: str):
        self.detail = {"operator": operator, "value": value}
        message = f"Expected a numeric value after '{operator}', got: '{value}'"
        super().__init__(message)

class InvalidValueFormatException(RulesetFactoriesException):
    def __init__(self, operator: str, value: str):
        self.detail = {"operator": operator, "value": value}
        message = f"Invalid value format '{value}' for operator '{operator}'"
        super().__init__(message)

class MissingIfKeywordException(RulesetFactoriesException):
    def __init__(self, rule_str: str):
        self.detail = {"rule": rule_str}
        message = f"Rule must start with 'IF': {rule_str}"
        super().__init__(message)

class MLRulesParsingException(RulesetFactoriesException):
    def __init__(self, original_exception: Exception):
        self.detail = {"original_exception": str(original_exception)}
        message = ("Error occurred during parsing of the MLRules model. "
                   "Ensure that the input data originates from the correct algorithm.")
        super().__init__(message)

class LordParsingException(RulesetFactoriesException):
    def __init__(self, original_exception: Exception):
        self.detail = {"original_exception": str(original_exception)}
        message = ("Error occurred during parsing of the LORD model. "
                   "Ensure that the input data originates from the correct algorithm.")
        super().__init__(message)