import unittest
import warnings

from decision_rules.filtering import filter_ruleset
from decision_rules.filtering._helpers import calculate_ruleset_prediction_score
from decision_rules.helpers.measures import get_measure_function_by_name
from tests.loaders import load_classification_dataset
from tests.loaders import load_classification_ruleset
from tests.loaders import load_iris_dataset
from tests.loaders import load_iris_ruleset
from tests.loaders import load_regression_dataset
from tests.loaders import load_regression_ruleset
from tests.loaders import load_survival_dataset
from tests.loaders import load_survival_ruleset


class TestRulesetFiltering(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.classification_ruleset = load_classification_ruleset()
        dataset = load_classification_dataset()
        self.classification_dataset = dataset.drop(
            "Salary", axis=1), dataset["Salary"]
        self.regression_ruleset = load_regression_ruleset()
        dataset = load_regression_dataset()
        self.regression_dataset = dataset.drop(
            "label", axis=1), dataset["label"]
        self.survival_ruleset = load_survival_ruleset()
        dataset = load_survival_dataset()
        dataset = dataset.drop(
            "survival_status", axis=1), dataset["survival_status"]
        self.survival_dataset = dataset
        self.iris_ruleset = load_iris_ruleset()
        dataset = load_iris_dataset()
        self.iris_dataset = dataset.drop("class", axis=1), dataset["class"]

    def base_test_filter_ruleset(self, ruleset, dataset, algorithm, loss, measure):
        if measure is not None:
            if isinstance(measure, str):
                measure_function = get_measure_function_by_name(measure)
            else:
                measure_function = measure
        else:
            measure_function = None
        coverage_matrix = ruleset.update(*dataset, measure_function)
        base_prediction_score = calculate_ruleset_prediction_score(
            ruleset, *dataset, coverage_matrix)
        if loss is None:
            loss = 0.0
        if base_prediction_score >= 0:
            target_score = base_prediction_score * (1 - loss)
        else:
            target_score = base_prediction_score * (1 + loss)
        filtered_ruleset = filter_ruleset(
            ruleset, *dataset, algorithm, loss, measure,
        )
        filtered_coverage_matrix = filtered_ruleset.calculate_coverage_matrix(
            dataset[0])
        filtered_prediction_score = calculate_ruleset_prediction_score(
            filtered_ruleset, *dataset, filtered_coverage_matrix)
        # additional check - filtering should not alter the original ruleset object
        base_prediction_score_check = calculate_ruleset_prediction_score(
            ruleset, *dataset, coverage_matrix)
        self.assertLessEqual(len(filtered_ruleset.rules),
                             len(ruleset.rules))
        self.assertGreaterEqual(filtered_prediction_score,
                                target_score)
        self.assertEqual(base_prediction_score, base_prediction_score_check)
