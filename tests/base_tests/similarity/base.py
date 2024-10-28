import unittest

from decision_rules.measures import precision
from decision_rules.problem import ProblemTypes
from tests.loaders import load_dataset
from tests.loaders import load_ruleset


class BaseSimilarityTest(unittest.TestCase):
    def setUp(self):
        self.ruleset1 = load_ruleset(
            "classification/salary.json", ProblemTypes.CLASSIFICATION)
        self.ruleset2 = load_ruleset(
            "classification/salary.json", ProblemTypes.CLASSIFICATION)
        self.dataset = load_dataset("classification/salary.csv")
        X, y = self.dataset.drop(["Salary"], axis=1), self.dataset["Salary"]
        self.ruleset1.update(X, y, precision)
        self.ruleset2.update(X, y, precision)
