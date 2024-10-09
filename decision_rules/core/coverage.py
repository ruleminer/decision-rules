"""
Contains rule coverage class
"""
from typing import TypedDict


class InvalidCoverageError(ValueError):
    pass


class Coverage:
    """Rule coverage"""

    p: int
    "Positive covered examples."

    n: int
    "Negative covered examples."

    P: int
    "All positive examples."

    N: int
    "All negative examples."

    def __init__(self, p: int, n: int, P: int, N: int):
        self.p = p
        self.n = n
        self.P = P
        self.N = N
        self._validate()

    def _validate(self):
        if any(e is None for e in self.as_tuple()):
            return
        if self.p > self.P:
            raise InvalidCoverageError('Invalid coverage: p is greater than P')
        if self.n > self.N:
            raise InvalidCoverageError('Invalid coverage: n is greater than N')

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.p, self.n, self.P, self.N)

    def __str__(self) -> str:
        return f'(p={self.p}, n={self.n}, P={self.P}, N={self.N})'


class ClassificationCoverageInfodict(TypedDict):
    p: int
    n: int
    P: int
    N: int


class RegressionCoverageInfodict(ClassificationCoverageInfodict):
    train_covered_y_std: float
    train_covered_y_mean: float


class SurvivalCoverageInfodict(ClassificationCoverageInfodict):
    median_survival_time: float
    median_sruvival_time_cli: float
    restricted_mean_survival_time: float
    events_count: int
    censored_count: int
    log_rank: float
    kaplan_meier_estimator: dict[str, list[float]]
