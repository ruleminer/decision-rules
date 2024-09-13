"""
Contains global configuration conditions classes.

Attributes:
    FLOAT_DISPLAY_PRECISION (str): controls floating points numbers precision
        when displaying rules as strings.
    CONCISE_NUMERICAL_CONDITIONS_FORM (bool): If set to true, string representations
        of numerical elementary conditions where one of the interval boundary is
        +/- infinity will be writen in more concise form of "X < right" or "X > left"
        instead of the full form: "X = <left, right)"
"""
FLOAT_DISPLAY_PRECISION: str = '2f'
CONCISE_NUMERICAL_CONDITIONS_FORM: bool = True
