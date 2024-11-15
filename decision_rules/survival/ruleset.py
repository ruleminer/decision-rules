"""
Contains survival ruleset class.
"""
from __future__ import annotations

from typing import Optional, Type, Union

import numpy as np
import pandas as pd

from decision_rules.core.coverage import SurvivalCoverageInfodict
from decision_rules.core.exceptions import InvalidStateError
from decision_rules.core.metrics import AbstractRulesMetrics
from decision_rules.core.prediction import PredictionStrategy
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.importances._survival.attributes import \
    SurvivalRuleSetAttributeImportances
from decision_rules.importances._survival.conditions import \
    SurvivalRuleSetConditionImportances
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator
from decision_rules.survival.metrics import SurvivalRulesMetrics
from decision_rules.survival.prediction import (BestRulePredictionStrategy,
                                                SurvivalPrediction,
                                                VotingPredictionStrategy)
from decision_rules.survival.rule import SurvivalConclusion, SurvivalRule


class SurvivalRuleSet(AbstractRuleSet):
    """Survival ruleset allowing to perform prediction on data
    """

    def __init__(
        self,
        rules: list[SurvivalRule],
        survival_time_attr: str,
    ) -> None:
        """
        Args:
            rules (list[SurvivalRule]):
        """
        self.rules: list[SurvivalRule]
        super().__init__(rules)
        self.survival_time_attr_name: str = survival_time_attr
        self.decision_attribute: str = (
            self.rules[0].conclusion.column_name
            if len(rules) > 0 else None
        )
        self.default_conclusion = SurvivalConclusion(
            value=None,
            column_name=self.decision_attribute
        )
        self._stored_default_conclusion = self.default_conclusion

    def _calculate_P_N(self, y_uniques: np.ndarray, y_values_count: np.ndarray):  # pylint: disable=invalid-name
        return

    def get_metrics_object_instance(self) -> AbstractRulesMetrics:
        return SurvivalRulesMetrics(self.rules)

    def update_using_coverages(
        self,
        coverages_info: dict[str, SurvivalCoverageInfodict],
        columns_names: list[str] = None,
        *args,
        **kwargs,
    ):
        for rule in self.rules:
            coverage_info: SurvivalCoverageInfodict = coverages_info[rule.uuid]
            rule.conclusion.estimator.update(
                coverage_info['kaplan_meier_estimator'],
                update_additional_indicators=True
            )
            rule.conclusion.value = coverage_info['median_survival_time']
            rule.conclusion.median_survival_time_ci_lower = coverage_info[
                'median_survival_time_ci_lower']
            rule.conclusion.median_survival_time_ci_upper = coverage_info[
                'median_survival_time_ci_upper']
            rule.conclusion.estimator.events_count = coverage_info['events_count']
            rule.conclusion.estimator.censored_count = coverage_info['censored_count']
            rule.log_rank = coverage_info['log_rank']

        super().update_using_coverages(coverages_info,
                                       KaplanMeierEstimator.log_rank, columns_names)

    def update(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        _measure=None
    ) -> np.ndarray:
        # the `measure` is always `log_rank` for survival rulesets,
        # but the parameter is kept for compatibility with other types
        if _measure is not None:
            raise ValueError(
                "The parameter `measure` should not be set for `SurvivalRuleSet` - `log_rank` will always be used.")

        if len(self.rules) == 0:
            raise ValueError(
                '"update" cannot be called on empty ruleset with no rules.'
            )

        if self.column_names is None:
            self.column_names = X_train.columns.tolist()
        # sort data by survival time
        survival_time_attr_index = self.column_names.index(
            self.survival_time_attr_name)
        X_train, y_train = self._sanitize_dataset(X_train, y_train)
        survival_time = X_train[:, survival_time_attr_index]
        sorted_indices = np.argsort(survival_time)
        survival_time_sorted = survival_time[sorted_indices]
        y_train_sorted = y_train[sorted_indices]
        X_train_sorted = X_train[sorted_indices, :]

        # fit Kaplan Meier estimator on whole dataset as default conclusion
        self.default_conclusion = SurvivalConclusion(
            value=None,
            column_name=self.decision_attribute
        )
        self.default_conclusion.estimator = KaplanMeierEstimator()
        self.default_conclusion.estimator.fit(
            survival_time_sorted,
            y_train_sorted,
            skip_sorting=True  # skip sorting (dataset is already sorted)
        )
        self.default_conclusion.value = self.default_conclusion.estimator.median_survival_time
        self._stored_default_conclusion = self.default_conclusion

        for rule in self.rules:
            rule.column_names = self.column_names
            rule.set_survival_time_attr(self.survival_time_attr_name)

        coverage_matrix: np.ndarray = self.calculate_rules_coverages(
            X_train_sorted,
            y_train_sorted,
            skip_sorting=True  # skip sorting (dataset is already sorted)
        )
        self.calculate_rules_weights(KaplanMeierEstimator.log_rank)

        reverted_sorted_indices = np.argsort(sorted_indices)
        return coverage_matrix[reverted_sorted_indices]

    def calculate_rules_metrics(
        self,
        X: pd.DataFrame,  # pylint: disable=invalid-name
        y: pd.Series,  # pylint: disable=invalid-name
        metrics_to_calculate: Optional[list[str]] = None
    ) -> dict[dict[str, str, float]]:
        for rule in self.rules:
            rule.premise.cached = True
        try:
            self.update(X, y)
            metrics: SurvivalRulesMetrics = self.get_metrics_object_instance()
            metrics_values: dict = metrics.calculate(
                X, y, metrics_to_calculate)
            return metrics_values
        except Exception as e:
            raise e
        finally:
            for rule in self.rules:
                rule.premise.invalidate_cache()

    def calculate_rules_weights(
        self,
        measure=None
    ):
        """
        Args:
            measure: quality measure function, in case of Survival it is always log_rank or if not specified, then voting_weight is 1

        Raises:
            ValueError: if any of the rules in ruleset has uncalculated coverage
        """
        if measure is None:
            for rule in self.rules:
                rule.voting_weight = 1
        else:
            for rule in self.rules:
                if rule.log_rank is None:
                    raise ValueError(
                        'Tried to calculate voting weight of a rule with uncalculated log_rank.' +
                        'You should either call `RuleSet.calculate_rules_coverages` method - to ' +
                        'calculate log_rank of all rules - or call `Rule.calculate_coverage` ' +
                        '- to calculate coverage of this specific rule'
                    )
                rule.voting_weight = rule.log_rank
        self._voting_weights_calculated = True

    def calculate_condition_importances(self, X: pd.DataFrame, y: pd.Series, *args) -> dict[str, float]:
        X, y = self._sanitize_dataset(X, y)
        condtion_importances_generator = SurvivalRuleSetConditionImportances(
            self)
        self.condition_importances = condtion_importances_generator.calculate_importances(
            X, y
        )
        return self.condition_importances

    def calculate_attribute_importances(self, condition_importances: dict[str, float]) -> dict[str, float]:
        attributes_importances_generator = SurvivalRuleSetAttributeImportances()
        self.attribute_importances = attributes_importances_generator.calculate_importances_base_on_conditions(
            condition_importances)
        return self.attribute_importances

    def local_explainability(self, x: pd.Series) -> tuple[list[str], SurvivalPrediction]:  # pylint: disable=invalid-name
        """Calculate local explainability of ruleset for given instance.

        Args:
            x (pd.Series): Instance to explain

        Returns:
            list: list of rules uuid's covering instance
            SurvivalPrediction: Kaplan-Meier estimate of examples covered by rules
        """
        X: np.ndarray = self._sanitize_dataset(x).reshape(1, -1)
        coverage_matrix: np.ndarray = self.calculate_coverage_matrix(X)
        prediction: SurvivalPrediction = self.predict_using_coverage_matrix(
            coverage_matrix
        )[0]

        rules_covering_instance = [
            rule.uuid for i, rule in enumerate(self.rules)
            if coverage_matrix[0][i]
        ]
        return rules_covering_instance, prediction

    def integrated_bier_score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        y_pred: Optional[np.ndarray] = None
    ) -> float:
        """Calculate Integrated Brier Score (IBS)

        Args:
            X (pd.DataFrame): dataset
            y (pd.Series): survival status column
            y_pred (Optional[np.ndarray], optional): Model predictions. If not
                provided, this method will perform prediction on the provided dataset.
                Defaults to None.

        Returns:
            float: Integrated Brier Score value
        """
        survival_times = X[self.survival_time_attr_name].to_numpy()
        survival_status = y.to_numpy()
        X, y = self._sanitize_dataset(X, y)

        if self._stored_default_conclusion is None:
            raise InvalidStateError(
                'Cannot calculate IBS without default conclusion. ' +
                'Maybe you forgot to call update(...) method?'
            )

        censoring_KM: KaplanMeierEstimator = self._stored_default_conclusion.estimator.reverse()

        censored_events_mask = survival_status == '0'
        info_list: list[_IBSInfo] = []
        prediction: np.ndarray = self.predict(X) if y_pred is None else y_pred
        zipped_data = zip(survival_times, censored_events_mask, prediction)
        for i, (time, is_censored, pred) in enumerate(zipped_data):
            km: KaplanMeierEstimator = SurvivalPrediction.to_kaplan_meier(pred)
            time: float = survival_times[i]
            info = _IBSInfo(time, is_censored, km)
            if km is not None:
                info_list.append(info)

        # sort info list by time
        sorted_info = sorted(info_list, key=lambda x: x.time, reverse=False)
        info_size = len(sorted_info)
        brier_score: list[float] = []

        prev_info = sorted_info[0]
        for i, info in enumerate(sorted_info):
            bt = info.time
            if (i > 0) and bt == prev_info.time:
                brier_score.append(prev_info.time)
            else:
                brier_sum = 0
                for si in sorted_info:
                    if si.time <= bt and not si.is_censored:
                        g = censoring_KM.get_probability_at(si.time)
                        if g > 0:
                            p = si.estimator.get_probability_at(bt)
                            brier_sum += (p * p) / g
                    elif si.time > bt:
                        g = censoring_KM.get_probability_at(bt)
                        if g > 0:
                            p = 1 - si.estimator.get_probability_at(bt)
                            brier_sum += (p * p) / g

                brier_score.append(brier_sum/info_size)
            prev_info = info

        diffs: list[float] = []
        diffs.append(sorted_info[0].time)
        for i in range(1, info_size):
            diffs.append(sorted_info[i].time - sorted_info[i - 1].time)

        score_sum = sum([
            diffs[i] * brier_score[i]
            for i in range(info_size)
        ])
        score = score_sum / sorted_info[info_size - 1].time

        return score

    def _sanitize_dataset(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        to_numpy: bool = True
    ) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
        res = super()._sanitize_dataset(X, y, to_numpy)
        if y is not None:
            y_sanitized: np.ndarray = res[1]
            self._validate_survival_status_column(y_sanitized)
        return res

    def _validate_survival_status_column(self, survival_status: np.ndarray):
        if set(np.unique(survival_status)) - {'0', '1'}:
            raise ValueError(
                'y (survival status) must be of string type and contain only "0" and "1" values.')

    def _map_prediction_values(self, predictions: np.ndarray) -> np.ndarray:
        return np.array([
            SurvivalPrediction.from_kaplan_meier(kaplan_meier)
            for kaplan_meier in predictions
        ])

    @property
    def prediction_strategies_choice(self) -> dict[str, Type[PredictionStrategy]]:
        return {
            'vote': VotingPredictionStrategy,
            'best_rule': BestRulePredictionStrategy,
        }

    def get_default_prediction_strategy_class(self) -> Type[PredictionStrategy]:
        return VotingPredictionStrategy


class _IBSInfo:
    def __init__(
        self,
        time: float,
        is_censored: bool,
        estimator: KaplanMeierEstimator,
    ) -> None:
        self.time = time
        self.is_censored = is_censored
        self.estimator = estimator
