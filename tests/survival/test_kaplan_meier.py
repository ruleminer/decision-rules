# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import json
import os
import unittest

import numpy as np
import pandas as pd
from decision_rules import measures
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import NominalCondition
from decision_rules.serialization.utils import JSONSerializer
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestKaplanMeierRuleSet(unittest.TestCase):

    def test_calculating_indicators(self):
        df = pd.read_csv(os.path.join(
            dir_path, '..', 'resources', 'survival', 'waltons.csv'
        ))

        survival_time = df['T']
        survival_status = df['E']

        kaplan_meier_estimator = KaplanMeierEstimator()
        kaplan_meier_estimator.fit(survival_time, survival_status.astype(str))

        decision_rules_median_survival_time = kaplan_meier_estimator.median_survival_time
        decision_rules_median_survival_time_cli_upper = kaplan_meier_estimator.median_survival_time_cli.iloc[
            0]["prob_upper_0.95"]
        decision_rules_median_survival_time_cli_lower = kaplan_meier_estimator.median_survival_time_cli.iloc[
            0]["prob_lower_0.95"]

        lifelines_median_survival_time = 56
        lifelines_median_survival_time_cli_upper = 58
        lifelines_median_survival_time_cli_lower = 53

        self.assertTrue(
            np.isclose(decision_rules_median_survival_time,
                       lifelines_median_survival_time, rtol=1e-5),
            'Median survival time should be the same as in lifelines'
        )
        self.assertTrue(
            np.isclose(decision_rules_median_survival_time_cli_upper,
                       lifelines_median_survival_time_cli_upper, rtol=1e-5),
            'Median survival time cli upper should be the same as in lifelines'
        )
        self.assertTrue(
            np.isclose(lifelines_median_survival_time_cli_lower,
                       lifelines_median_survival_time_cli_lower, rtol=1e-5),
            'Median survival time cli lower should be the same as in lifelines'
        )

    def test_estimator_fit(self):
        df = pd.read_csv(os.path.join(
            dir_path, '..', 'resources', 'survival', 'BHS.csv'
        ))

        survival_time = df['survival_time'].to_numpy()
        survival_status = df['survival_status'].to_numpy()

        kaplan_meier_estimator = KaplanMeierEstimator()
        kaplan_meier_estimator.fit(
            survival_time, survival_status.astype(int).astype(str))

        rulekit_kaplan_meier = {'times': [1.1,  2.7,  4.3,  4.4,  6.4,  8.4,  9.9, 13.2, 16.1, 17.4, 17.8,
                                18.2, 19.1, 19.9, 21.1, 22.1, 22.6, 25.1, 26.1, 26.9, 28.1],
                                'events_count': [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
                                'at_risk_count': [75, 74, 73, 72, 71, 70, 69, 68, 67, 47, 46, 45, 44, 36, 35, 34, 22,
                                                  21, 15, 14, 13],
                                'probabilities': [0.98666667, 0.97333333, 0.96, 0.94666667, 0.93333333,
                                                  0.92, 0.90666667, 0.89333333, 0.89333333, 0.87432624,
                                                  0.85531915, 0.83631206, 0.83631206, 0.81308117, 0.78985028,
                                                  0.76661939, 0.73177305, 0.73177305, 0.68298818, 0.63420331,
                                                  0.58541844]}
        decision_rules_kaplan_meier = kaplan_meier_estimator.get_dict()
        decision_rules_kaplan_meier.pop("censored_count")
        decision_rules_kaplan_meier["probabilities"] = [
            round(x, 8) for x in decision_rules_kaplan_meier["probabilities"]]
        self.assertTrue(
            all([
                np.isclose(
                    decision_rules_kaplan_meier[key], rulekit_kaplan_meier[key], rtol=1e-5).all()
                for key in decision_rules_kaplan_meier.keys()
            ]),
            'Kaplan Meier should be the same as in rulekit'
        )


if __name__ == '__main__':
    unittest.main()
