# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from decision_rules import measures
from decision_rules.problem import ProblemTypes
from tests.loaders import load_dataset
from tests.loaders import load_ruleset


class TestUpdateRuleSetMetaBase(unittest.TestCase):
    def _test_update_meta(
            self,
            problem_type: ProblemTypes,
            dataset_path: str,
            ruleset_path: str,
            target_column: str,
            measure: Optional[str],
    ):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        dataset = load_dataset(dataset_path)
        X, y = dataset.drop(columns=target_column), dataset[target_column]
        new_attributes = X.columns.tolist()
        ruleset = load_ruleset(ruleset_path, problem_type)
        if problem_type == ProblemTypes.SURVIVAL:
            y = y.astype("int").astype("str")
        coverage_matrix_before = ruleset.update(X, y, measure)
        prediction_before = ruleset.predict(X)
        ruleset.update_meta(new_attributes)
        coverage_matrix_after = ruleset.update(X, y, measure)
        prediction_after = ruleset.predict(X)
        self.assertTrue(np.array_equal(
            coverage_matrix_before, coverage_matrix_after))
        if problem_type == ProblemTypes.SURVIVAL:
            for row1, row2 in zip(prediction_before, prediction_after):
                self.assertTrue(pd.DataFrame(row1).equals(pd.DataFrame(row2)))
        else:
            self.assertTrue(np.array_equal(
                prediction_before, prediction_after))


class TestUpdateClassificationRuleSetMeta(TestUpdateRuleSetMetaBase):
    def test_update_meta(self):
        self._test_update_meta(
            ProblemTypes.CLASSIFICATION,
            "classification/diabetes.csv",
            "classification/diabetes_half.json",
            "class",
            measures.precision,
        )


class TestUpdateRegressionRuleSetMeta(TestUpdateRuleSetMetaBase):
    def test_update_meta(self):
        self._test_update_meta(
            ProblemTypes.REGRESSION,
            "regression/boston.csv",
            "regression/boston_half.json",
            "MEDV",
            measures.correlation,
        )


class TestUpdateSurvivalRuleSetMeta(TestUpdateRuleSetMetaBase):
    def test_update_meta(self):
        self._test_update_meta(
            ProblemTypes.SURVIVAL,
            "survival/bone_marrow.csv",
            "survival/bone_marrow_half.json",
            "survival_status",
            None,
        )
