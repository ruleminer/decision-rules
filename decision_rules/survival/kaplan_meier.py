from __future__ import annotations

from bisect import bisect_left
from typing import Optional
from typing import TypedDict
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import norm


class KaplanMeierEstimatorDict(TypedDict):
    times: np.ndarray
    events_count: np.ndarray
    censored_count: np.ndarray
    at_risk_count: np.ndarray
    probabilities: np.ndarray


class SurvInfo:
    def __init__(
        self,
        time: np.ndarray,  # int,
        events_count: np.ndarray,  # int,
        censored_count: np.ndarray,  # int,
        at_risk_count: np.ndarray,  # int,
        probability: np.ndarray,  # float,
    ) -> None:
        self.time: np.ndarray = time
        self.events_count: np.ndarray = events_count
        self.censored_count: np.ndarray = censored_count
        self.at_risk_count: np.ndarray = at_risk_count
        self.probability: np.ndarray = probability
        self.sq: np.ndarray = np.zeros(shape=len(time))


class KaplanMeierEstimator():

    def __init__(
        self,
        surv_info: Optional[SurvInfo] = None
    ) -> None:
        self.median_survival_time: float = None
        self.median_survival_time_cli: float = None
        self.restricted_mean_survival_time: float = None
        self.interval: float = None

        self.times: np.ndarray = np.array([])
        self.events_counts: np.ndarray = np.array([])
        self.censored_counts: np.ndarray = np.array([])
        self.at_risk_counts: np.ndarray = np.array([])
        self.probabilities: np.ndarray = np.array([])

        if surv_info is None:
            self._surv_info: SurvInfo = SurvInfo(
                time=np.array([]),
                events_count=np.array([]),
                censored_count=np.array([]),
                at_risk_count=np.array([]),
                probability=np.array([])
            )
        else:
            self._surv_info: SurvInfo = surv_info
            self.process_surv_info(self._surv_info)
            self._update_additional_indicators()

    @property
    def surv_info(self) -> SurvInfo:
        return self._surv_info

    @surv_info.setter
    def surv_info(self, surv_info: SurvInfo):
        self._surv_info = surv_info
        self.process_surv_info(surv_info)

    def process_surv_info(self, surv_info: SurvInfo):
        self.times = surv_info.time
        self.events_counts = surv_info.events_count
        self.censored_counts = surv_info.censored_count
        self.at_risk_counts = surv_info.at_risk_count
        self.probabilities = surv_info.probability

        self.len_of_times = len(surv_info.time)
        self.events_count_sum = np.sum(surv_info.events_count)
        self.censored_count_sum = np.sum(surv_info.censored_count)

    def update(self, kaplan_meier_estimator_dict: dict[str, list[float]], update_additional_indicators: bool = False) -> KaplanMeierEstimator:
        kaplan_meier_estimator_dict: KaplanMeierEstimatorDict = {
            k: np.array(v)
            for k, v in kaplan_meier_estimator_dict.items()
        }
        self.surv_info = SurvInfo(
            time=kaplan_meier_estimator_dict["times"],
            events_count=kaplan_meier_estimator_dict["events_count"],
            censored_count=kaplan_meier_estimator_dict["censored_count"],
            at_risk_count=kaplan_meier_estimator_dict["at_risk_count"],
            probability=kaplan_meier_estimator_dict["probabilities"],
        )
        self.process_surv_info(self.surv_info)
        if update_additional_indicators:
            self._update_additional_indicators()
        return self

    def _update_additional_indicators(self):
        self.interval = self.calculate_interval()
        self.median_survival_time, self.median_survival_time_cli = self.calcualte_indicators()

    def fit(self, survival_time: np.ndarray, survival_status:  np.ndarray, skip_sorting: bool = False) -> KaplanMeierEstimator:
        """Fit Kaplan Meier estimator on given data

        Args:
            survival_time (np.ndarray): survival time data
            survival_status (np.ndarray): survival status data
            skip_sorting (bool, optional): Flag allowing to optionally skip sorting based
                on survival time. It could be used to speed up the computation if the provided
                data is already sorted ascending by survival time. Defaults to False (this method
                will sort the data under the hood).

        Returns:
            KaplanMeierEstimator: fitted estimator
        """
        if survival_time.shape[0] == 0:
            return self

        if not skip_sorting:
            # sort surv_info_list by survival_time
            sorted_indices = np.argsort(survival_time)
            survival_time = survival_time[sorted_indices]
            survival_status = survival_status[sorted_indices]

        events_ocurences: np.ndarray = survival_status == '1'
        censored_ocurences = np.logical_not(events_ocurences).astype(int)
        events_ocurences = events_ocurences.astype(int)

        events_counts = events_ocurences
        censored_counts = censored_ocurences
        at_risk_count = np.zeros(shape=survival_time.shape)

        at_risk_count = survival_time.shape[0]
        grouped_data = {
            'events_count': {},
            'censored_count': {},
            'at_risk_count': {},
        }
        time_point_prev = survival_time[0]
        for (time_point, event_count, censored_count) in zip(survival_time, events_counts, censored_counts):
            if time_point != time_point_prev:
                grouped_data['at_risk_count'][time_point_prev] = at_risk_count
                at_risk_count -= grouped_data['events_count'][time_point_prev]
                at_risk_count -= grouped_data['censored_count'][time_point_prev]
                time_point_prev = time_point
            if time_point in grouped_data['events_count']:
                grouped_data['events_count'][time_point] += event_count
                grouped_data['censored_count'][time_point] += censored_count
            else:
                grouped_data['events_count'][time_point] = event_count
                grouped_data['censored_count'][time_point] = censored_count

        grouped_data['at_risk_count'][time_point] = at_risk_count

        unique_times = np.array(list(grouped_data['events_count'].keys()))
        events_count = np.array(list(grouped_data['events_count'].values()))
        censored_count = np.array(
            list(grouped_data['censored_count'].values()))
        at_risk_count = np.array(list(grouped_data['at_risk_count'].values()))

        surv_info = SurvInfo(
            time=unique_times,
            events_count=events_count,
            censored_count=censored_count,
            at_risk_count=at_risk_count,
            probability=np.zeros(shape=unique_times.shape),
        )

        surv_info = self.calculate_probabilities(surv_info)
        self.surv_info = surv_info
        self._update_additional_indicators()
        return self

    def calculate_probabilities(self, surv_info: SurvInfo) -> SurvInfo:
        surv_info.probability = np.zeros(shape=surv_info.at_risk_count.shape)
        non_zero_probability_mask = (surv_info.at_risk_count != 0)

        masked_at_risk_count = surv_info.at_risk_count[non_zero_probability_mask]
        events_count = surv_info.events_count[non_zero_probability_mask]

        surv_info.probability[non_zero_probability_mask] = (
            (masked_at_risk_count - events_count) / masked_at_risk_count
        )
        surv_info.probability = np.cumprod(surv_info.probability)
        return surv_info

    def calculate_interval(self) -> pd.DataFrame:
        with np.errstate(divide="ignore", invalid='ignore'):
            tmp = (self.events_counts / (self.at_risk_counts *
                   (self.at_risk_counts - self.events_counts)))
        cumulative_sq_ = np.cumsum(np.nan_to_num(tmp, posinf=0, neginf=0))
        return self.calculate_bounds(self.times, self.probabilities, cumulative_sq_)

    def calcualte_indicators(self) -> tuple[float, float]:
        survival_function = pd.DataFrame(index=self.times)
        survival_function["KM_estimate"] = self.probabilities
        median_survival_time = self.calculate_median_survival_time(
            survival_function)
        median_survival_time_cli = self.calculate_median_survival_time(
            self.interval)
        return median_survival_time, median_survival_time_cli

    def calculate_median_survival_time(self, survival_function: pd.DataFrame) -> Union[float, pd.DataFrame]:
        return self.qth_survival_times(0.5, survival_function)

    def qth_survival_times(self, q: float, survival_functions: pd.DataFrame) -> Union[float, pd.DataFrame]:
        """
        Find the times when one or more survival functions reach the qth percentile.
        """
        # pylint: disable=cell-var-from-loop,misplaced-comparison-constant,no-else-return
        q = self._to_1d_array(q)
        q = pd.Series(q.reshape(q.size), dtype=float)

        if not ((q <= 1).all() and (0 <= q).all()):
            raise ValueError("q must be between 0 and 1")

        if survival_functions.shape[1] == 1 and q.shape == (1,):
            q = q[0]
            return survival_functions.apply(lambda s: self.qth_survival_time(q, s)).iloc[0]
        else:
            d = {_q: survival_functions.apply(
                lambda s: self.qth_survival_time(_q, s)) for _q in q}
            survival_times = pd.DataFrame(d).T
            if q.duplicated().any():
                survival_times = survival_times.loc[q]

            return survival_times

    def qth_survival_time(self, q: float, survival_function: Union[pd.DataFrame, pd.Series]) -> float:
        """
        Returns the time when a single survival function reaches the qth percentile, that is,
        solves  :math:`q = S(t)` for :math:`t`.
        """
        if isinstance(survival_function, pd.DataFrame):
            if survival_function.shape[1] > 1:
                raise ValueError(
                    "Expecting a DataFrame (or Series) with a single column. Provide that or use utils.qth_survival_times."
                )
            return self.qth_survival_time(q, survival_function.T.squeeze())
        elif isinstance(survival_function, pd.Series):
            if survival_function.iloc[-1] > q:
                return np.inf
            return survival_function.index[(-survival_function).searchsorted([-q])[0]]
        else:
            raise ValueError(
                "Unable to compute median of object %s - should be a DataFrame, Series or lifelines univariate model"
                % survival_function
            )

    def _to_1d_array(self, x) -> np.ndarray:
        v = np.atleast_1d(x)
        try:
            if v.shape[0] > 1 and v.shape[1] > 1:
                raise ValueError("Wrong shape (2d) given to _to_1d_array")
        except IndexError:
            pass
        return v

    def calculate_bounds(self, times: np.array, probabilities: np.array, cumulative_sq: np.array, alpha=0.05) -> pd.DataFrame:
        # This method calculates confidence intervals using the exponential Greenwood formula.
        # See https://www.math.wustl.edu/%7Esawyer/handouts/greenwood.pdf
        z = norm.ppf(1 - alpha / 2)
        df = pd.DataFrame(index=times)
        with np.errstate(divide="ignore", invalid='ignore'):
            v = np.array(np.log(probabilities)).reshape(-1, 1)
            cumulative_sq_ = np.array(cumulative_sq).reshape(-1, 1)

            ci_labels = ["%s_lower_%g" %
                         ("prob", 1 - alpha), "%s_upper_%g" % ("prob", 1 - alpha)]

            df[ci_labels[0]] = np.exp(-np.exp(np.log(-v) -
                                              z * np.sqrt(cumulative_sq_) / v))
            df[ci_labels[1]] = np.exp(-np.exp(np.log(-v) +
                                              z * np.sqrt(cumulative_sq_) / v))
        return df.fillna(1.0)

    @staticmethod
    def average(estimators: list[KaplanMeierEstimator]) -> KaplanMeierEstimator:
        unique_times: np.ndarray = np.unique(
            np.concatenate([estimator.times for estimator in estimators])
        )
        unique_times.sort()

        number_of_estimators = len(estimators)
        probabilities = np.zeros(shape=unique_times.shape)
        for i, time in enumerate(unique_times):
            probabilities_sum = sum([
                estimator.get_probability_at(time)
                for estimator in estimators
            ])
            probabilities[i] = probabilities_sum / number_of_estimators

        surv_info = SurvInfo(
            time=unique_times,
            events_count=np.zeros(shape=unique_times.shape),
            censored_count=np.zeros(shape=unique_times.shape),
            at_risk_count=np.zeros(shape=unique_times.shape),
            probability=probabilities
        )
        avg_estimator = KaplanMeierEstimator()
        avg_estimator.surv_info = surv_info
        avg_estimator.interval = avg_estimator.calculate_interval()
        avg_estimator.median_survival_time, avg_estimator.median_survival_time_cli = avg_estimator.calcualte_indicators()
        return avg_estimator

    def binary_search(self, arr, target):
        index = bisect_left(arr, target)
        if index < self.len_of_times and arr[index] == target:
            return index
        else:
            return (-index - 1)

    def get_probability_at(self, time: int) -> float:
        index = self.binary_search(
            self.times, time)

        if index >= 0:
            return self.probabilities[index]

        index = ~index

        if index == 0:
            return 1

        return self.probabilities[index - 1]

    def get_events_count_at(self, time: int) -> int:
        index = self.binary_search(
            self.times, time)
        if index >= 0:
            return self.events_counts[index]

        return 0

    def get_at_risk_count_at(self, time: int) -> int:
        index = self.binary_search(
            self.times, time)
        if index >= 0:
            return self.at_risk_counts[index]

        index = ~index

        n = self.len_of_times
        if index == n:
            return self.at_risk_counts[n - 1]

        return self.at_risk_counts[index]

    def reverse(self) -> KaplanMeierEstimator:
        surv_info_rev = SurvInfo(
            time=np.copy(self.surv_info.time),
            # notice how events_count and censored_count are switche
            events_count=np.copy(self.surv_info.censored_count),
            censored_count=np.copy(self.surv_info.events_count),
            at_risk_count=np.copy(self.surv_info.at_risk_count),
            probability=np.zeros(shape=self.times.shape),
        )
        surv_info_rev = self.calculate_probabilities(surv_info_rev)
        rev_km = KaplanMeierEstimator(surv_info=surv_info_rev)
        return rev_km

    @staticmethod
    def compare_estimators(kme1: KaplanMeierEstimator, kme2: KaplanMeierEstimator) -> dict(str, float):
        results = dict()

        if (len(kme1.times) == 0) or (len(kme2.times) == 0):
            results["stats"] = 0
            results["p_value"] = 0
            return results

        times = set(kme1.times)
        times.update(kme2.times)

        x = 0
        y = 0

        for time in times:
            m1 = kme1.get_events_count_at(time)
            n1 = kme1.get_at_risk_count_at(time)

            m2 = kme2.get_events_count_at(time)
            n2 = kme2.get_at_risk_count_at(time)

            m = m1 + m2
            e2 = (n2 / (n1 + n2)) * m
            n = n1 + n2
            n_2 = n * n

            x += m2 - e2
            if (n_2 * (n - 1)) == 0:
                y += 0
            else:
                y += (n1 * n2 * m * (n - m1 - m2)) / (n_2 * (n - 1))

        results["stats"] = (x * x) / y
        results["p_value"] = 1 - chi2.cdf(results["stats"], 1)

        return results

    def get_dict(self) -> KaplanMeierEstimatorDict:
        return KaplanMeierEstimatorDict({
            "times": self.times.tolist(),
            "events_count": self.events_counts.tolist(),
            "censored_count": self.censored_counts.tolist(),
            "at_risk_count": self.at_risk_counts.tolist(),
            "probabilities": self.probabilities.tolist(),
        })

    @staticmethod
    def log_rank(survival_time: np.ndarray, survival_status:  np.ndarray, covered_examples: np.ndarray, uncovered_examples: np.ndarray) -> float:  # pylint: disable=missing-function-docstring
        covered_estimator = KaplanMeierEstimator().fit(
            survival_time[covered_examples], survival_status[covered_examples])
        uncovered_estimator = KaplanMeierEstimator().fit(
            survival_time[uncovered_examples], survival_status[uncovered_examples])

        stats_and_pvalue = KaplanMeierEstimator().compare_estimators(
            covered_estimator, uncovered_estimator)

        return 1 - stats_and_pvalue["p_value"]
