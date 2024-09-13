from enum import Enum


class ProblemTypes(str, Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    SURVIVAL = 'survival'
