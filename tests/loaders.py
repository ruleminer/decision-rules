import json
import os
from typing import Tuple, Optional

import pandas as pd
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.problem import ProblemTypes
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.serialization import JSONSerializer
from decision_rules.survival.ruleset import SurvivalRuleSet


def deserialize_ruleset(ruleset: dict, problem_type: ProblemTypes) -> ClassificationRuleSet:
    PROBLEM_TYPE_MAPPING = {
        ProblemTypes.CLASSIFICATION: ClassificationRuleSet,
        ProblemTypes.REGRESSION: RegressionRuleSet,
        ProblemTypes.SURVIVAL: SurvivalRuleSet
    }
    return JSONSerializer.deserialize(
        ruleset,
        PROBLEM_TYPE_MAPPING[problem_type]
    )


def load_ruleset(path: str, problem_type: ProblemTypes) -> AbstractRuleSet:
    ruleset_file_path: str = os.path.join(load_resources_path(), path)
    with open(ruleset_file_path, 'r', encoding='utf-8') as file:
        return deserialize_ruleset(json.load(file), problem_type)


def load_classification_ruleset() -> ClassificationRuleSet:
    return load_ruleset("classification/salary.json", ProblemTypes.CLASSIFICATION)


def load_iris_ruleset() -> ClassificationRuleSet:
    return load_ruleset("iris_ruleset.json", ProblemTypes.CLASSIFICATION)


def load_regression_ruleset() -> RegressionRuleSet:
    return load_ruleset("regression/diabetes_ruleset.json", ProblemTypes.REGRESSION)


def load_survival_ruleset() -> SurvivalRuleSet:
    return load_ruleset("survival/BHS_ruleset.json", ProblemTypes.SURVIVAL)


def load_dataset(path: str) -> pd.DataFrame:
    dataset_file_path: str = os.path.join(load_resources_path(), path)
    return pd.read_csv(dataset_file_path)


def load_dataset_to_x_y(path: str, y_col: str = "class") -> Tuple[pd.DataFrame, pd.Series]:
    """Read dataset from path relative to tests/resources and return X and y based 
    on y_col column name."""
    dataset = load_dataset(path)
    y = dataset[y_col]
    X = dataset.drop(y_col, axis=1)
    return X, y


def load_classification_dataset():
    return load_dataset("classification/salary.csv")


def load_iris_dataset():
    return load_dataset("iris.csv")


def load_regression_dataset():
    return load_dataset("regression/diabetes.csv")


def load_survival_dataset():
    dataset = load_dataset("survival/BHS.csv")
    dataset["survival_status"] = dataset["survival_status"].astype(
        "int").astype("str")
    return dataset


def load_resources_path() -> str:
    """Return path to resources directory"""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, 'resources')


def load_ruleset_factories_resources_path() -> str:
    """Return path to resources directory for ruleset_factories extra."""
    return os.path.join(load_resources_path(), 'ruleset_factories')
