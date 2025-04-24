import unittest
from functools import reduce

import pandas as pd

from decision_rules.classification import ClassificationRuleSet
from decision_rules.core.condition import AbstractCondition
from decision_rules.helpers.dataset_transformer import \
    ConditionalDatasetTransformer
from decision_rules.problem import ProblemTypes
from tests.loaders import load_dataset, load_ruleset


class TestConditionalDatasetTransformer(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.ruleset: ClassificationRuleSet = load_ruleset(
            "classification/heart_c_ruleset.json", ProblemTypes.CLASSIFICATION
        )
        df = load_dataset("classification/heart-c.csv")
        self.X, self.y = self.ruleset.split_dataset(df)

    def test_transform_top_level(self):
        conditions: list[AbstractCondition] = [r.premise for r in self.ruleset.rules]
        t = ConditionalDatasetTransformer(conditions)
        X_t: pd.DataFrame = t.transform(
            self.X.to_numpy(), self.X.columns, method="top_level"
        )

        self.assertEqual(X_t.shape[0], self.X.shape[0], "Should have as many rows as X")
        self.assertEqual(
            X_t.shape[1],
            len(set([r.premise for r in self.ruleset.rules])),
            "Should have as many columns as conditions passed to the transformer",
        )

    def test_transform_split(self):
        conditions: list[AbstractCondition] = [r.premise for r in self.ruleset.rules]
        t = ConditionalDatasetTransformer(conditions)
        X_t: pd.DataFrame = t.transform(
            self.X.to_numpy(), self.X.columns, method="split"
        )

        self.assertEqual(X_t.shape[0], self.X.shape[0], "Should have as many rows as X")
        expected_columns_count: int = len(
            set(reduce(lambda s, e: s.union(e.subconditions), conditions, set()))
        ) + len(conditions)
        self.assertEqual(
            X_t.shape[1],
            expected_columns_count,
            "Should have as many columns as conditions passed to the transformer"
            + "and their subconditions",
        )

    def test_transform_nested(self):
        conditions: list[AbstractCondition] = [r.premise for r in self.ruleset.rules]
        t = ConditionalDatasetTransformer([r.premise for r in self.ruleset.rules])
        X_t: pd.DataFrame = t.transform(
            self.X.to_numpy(), self.X.columns, method="nested"
        )

        self.assertEqual(X_t.shape[0], self.X.shape[0], "Should have as many rows as X")

        all_conditions_recursive: set[AbstractCondition] = set()

        def add_condition(
            conditions: set[AbstractCondition], condition: AbstractCondition
        ):
            conditions.add(condition)
            for subcondition in condition.subconditions:
                add_condition(conditions, subcondition)

        for condition in conditions:
            add_condition(all_conditions_recursive, condition)
        expected_columns_count: int = len(all_conditions_recursive)

        self.assertEqual(
            X_t.shape[1],
            expected_columns_count,
            "Should have as many columns as conditions passed to the transformer"
            + "and their subconditions recursive",
        )
