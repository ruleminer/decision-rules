from __future__ import annotations

from collections import namedtuple
from typing import TypedDict

import numpy as np
from scipy.stats import chi2

from decision_rules.survival.kaplan_meier import KaplanMeierEstimator, SurvInfo

_PrecalculatedEntry = namedtuple(
    "_PrecalculatedEntry", ["time_index", "is_exact_match"]
)


class _LogRankResult(TypedDict):
    p_value: float
    stats: float


class PrecalculatedKaplanMeierEstimator(KaplanMeierEstimator):
    """Kaplan-Meier estimator which precalculated function value for given times.
    It is still fitted only on the times passed to the fit method but allows to pass
    additional array of times to precalculate. This eliminates the usage of
    bisection algorithm when comparing two estimators which greatly improves "log_rank"
    calculation times.
    """

    def __init__(self, surv_info=None):
        super().__init__(surv_info)

        self._precalculated_entries: dict[float, _PrecalculatedEntry] = {}

    def fit(
        self,
        survival_time: np.ndarray,
        survival_status: np.ndarray,
        times_to_precalculate: np.ndarray,
        skip_sorting: bool = False,
    ) -> KaplanMeierEstimator:
        self._precalculated_entries = {}

        if survival_time.shape[0] == 0:
            return self

        if not skip_sorting:
            # sort surv_info_list by survival_time
            sorted_indices = np.argsort(survival_time)
            survival_time = survival_time[sorted_indices]
            survival_status = survival_status[sorted_indices]

        events_ocurences: np.ndarray = survival_status == "1"
        censored_ocurences = np.logical_not(events_ocurences).astype(int)
        events_ocurences = events_ocurences.astype(int)

        events_counts = events_ocurences
        censored_counts = censored_ocurences
        at_risk_count = np.zeros(shape=survival_time.shape)

        at_risk_count = survival_time.shape[0]
        grouped_data = {
            "events_count": {},
            "censored_count": {},
            "at_risk_count": {},
        }
        time_point_prev = survival_time[0]

        for time_point, event_count, censored_count in zip(
            survival_time, events_counts, censored_counts
        ):
            if time_point != time_point_prev:
                grouped_data["at_risk_count"][time_point_prev] = at_risk_count
                at_risk_count -= grouped_data["events_count"][time_point_prev]
                at_risk_count -= grouped_data["censored_count"][time_point_prev]
                time_point_prev = time_point
            if time_point in grouped_data["events_count"]:
                grouped_data["events_count"][time_point] += event_count
                grouped_data["censored_count"][time_point] += censored_count
            else:
                grouped_data["events_count"][time_point] = event_count
                grouped_data["censored_count"][time_point] = censored_count

        grouped_data["at_risk_count"][time_point] = at_risk_count

        unique_times = np.array(list(grouped_data["events_count"].keys()))
        events_count = np.array(list(grouped_data["events_count"].values()))
        censored_count = np.array(list(grouped_data["censored_count"].values()))
        at_risk_count = np.array(list(grouped_data["at_risk_count"].values()))

        self._precalculates_times(unique_times, times_to_precalculate)

        self.surv_info = SurvInfo(
            time=unique_times,
            events_count=events_count,
            censored_count=censored_count,
            at_risk_count=at_risk_count,
            probability=np.zeros(shape=unique_times.shape),
        )

        return self

    def _precalculates_times(
        self, unique_times: np.ndarray, times_to_precalculate: np.ndarray
    ):
        """Precalculates times indices for both times we are fitting on, and given times
        to precalculate. This "precalculation" calculates index at which given time
        would be inserted to preserve original ascending order. This method also check
        if given time was or wasn't present in the `unique_times` array.

        Args:
            unique_times (np.ndarray): All unique times model is fitted on.
            times_to_precalculate (np.ndarray): Times for which indices should be
                precalculated.
        """
        other_time_index: int = 0
        other_time: float = times_to_precalculate[other_time_index]
        max_time: float = float("-inf")

        # for all time to precalculate we find index at which it would be inserted to
        # preserve ascending order.
        for time_index, time in enumerate(unique_times):
            # insert all times lower than this time
            while other_time < time:
                self._precalculated_entries[other_time] = _PrecalculatedEntry(
                    time_index=time_index, is_exact_match=other_time == max_time
                )
                other_time_index += 1
                if other_time_index >= len(times_to_precalculate):
                    break
                other_time = times_to_precalculate[other_time_index]
            max_time = time

        # times we are fitting on are also inserted to precalculated entries dict
        for time_index, time in enumerate(unique_times):
            self._precalculated_entries[time] = _PrecalculatedEntry(
                time_index=time_index, is_exact_match=True
            )

        # at the end we add all times which are higher or equal to the max(unique_times)
        max_time = unique_times[-1]
        max_time_index = len(unique_times) - 1
        for other_time in times_to_precalculate[other_time_index:]:
            if other_time > max_time:
                self._precalculated_entries[other_time] = _PrecalculatedEntry(
                    time_index=-1,  # we use -1 for times higher than max time
                    is_exact_match=False,
                )
            else:
                self._precalculated_entries[other_time] = _PrecalculatedEntry(
                    max_time_index,  # we use last index value for time equal max time
                    is_exact_match=True,
                )

    def _get_precalculated_entry(self, time: float) -> _PrecalculatedEntry:
        return self._precalculated_entries[time]

    @staticmethod
    def compare_estimators(
        kme1: PrecalculatedKaplanMeierEstimator, kme2: PrecalculatedKaplanMeierEstimator
    ) -> _LogRankResult:

        if (len(kme1.times) == 0) or (len(kme2.times) == 0):
            return _LogRankResult(stats=0.0, p_value=0.0)

        times = set(np.concatenate([kme1.times, kme2.times]))

        x = 0
        y = 0

        for time in times:
            e: _PrecalculatedEntry = kme1._get_precalculated_entry(time)
            m1 = kme1.events_counts[e.time_index] if e.is_exact_match else 0
            n1 = kme1.at_risk_counts[e.time_index]

            e: _PrecalculatedEntry = kme2._get_precalculated_entry(time)
            m2 = kme2.events_counts[e.time_index] if e.is_exact_match else 0
            n2 = kme2.at_risk_counts[e.time_index]

            m = m1 + m2
            e2 = (n2 / (n1 + n2)) * m
            n = n1 + n2
            n_2 = n * n

            x += m2 - e2
            if (n_2 * (n - 1)) == 0:
                y += 0
            else:
                y += (n1 * n2 * m * (n - m1 - m2)) / (n_2 * (n - 1))

        statistic: float = (x * x) / y
        p_value: float = 1 - chi2.cdf(statistic, 1)
        return _LogRankResult(stats=statistic, p_value=p_value)

    @staticmethod
    def log_rank(
        survival_time: np.ndarray,
        survival_status: np.ndarray,
        covered_examples: np.ndarray,
        uncovered_examples: np.ndarray,
        skip_sorting: bool = False,
    ) -> float:  # pylint: disable=missing-function-docstring
        covered_times: np.ndarray = survival_time[covered_examples]
        uncovered_times: np.ndarray = survival_time[uncovered_examples]

        covered_estimator = PrecalculatedKaplanMeierEstimator().fit(
            covered_times,
            survival_status[covered_examples],
            times_to_precalculate=uncovered_times,
            skip_sorting=skip_sorting,
        )
        uncovered_estimator = PrecalculatedKaplanMeierEstimator().fit(
            uncovered_times,
            survival_status[uncovered_examples],
            times_to_precalculate=covered_times,
            skip_sorting=skip_sorting,
        )

        stats_and_pvalue = PrecalculatedKaplanMeierEstimator().compare_estimators(
            covered_estimator, uncovered_estimator
        )

        return 1 - stats_and_pvalue["p_value"]
