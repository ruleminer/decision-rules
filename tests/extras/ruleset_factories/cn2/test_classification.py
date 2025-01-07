# pylint: disable=missing-module-docstring, missing-class-docstring, missing-function-docstring
import unittest
import numpy as np

from Orange.classification.rules import CN2Classifier, CN2Learner
from Orange.data import Domain, Table, DiscreteVariable, ContinuousVariable

from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.ruleset_factories._factories.classification.cn2_factory import get_orange_cn2_factory_class
from decision_rules.serialization import JSONSerializer
from decision_rules.core.prediction import FirstRuleCoveringStrategy

from tests.loaders import load_dataset_to_x_y


class TestOrangeCN2ClassificationRuleSet(unittest.TestCase):
    """
    Tests for the ClassificationRuleSet produced by the Orange CN2 factory.
    Uses the same "deals-train.csv" dataset as the RuleKit tests.
    """

    cn2_model: CN2Classifier
    dataset_path: str = "classification/deals-train.csv"

    @classmethod
    def setUpClass(cls):
        """
        Loads X and y from the CSV dataset, converts them to an Orange Table,
        and trains a CN2Classifier.
        """
        # Load dataset into pandas
        X, y = load_dataset_to_x_y(cls.dataset_path)

        # Build an Orange Domain for the features and the class
        domain = Domain(
            [ContinuousVariable(name) for name in X.columns],
            DiscreteVariable(y.name, values=list(map(str, y.unique())))
        )
        table = Table.from_numpy(domain, X.values, y.values)

        # Train the CN2 model
        cn2_learner = CN2Learner()
        cn2_model = cn2_learner(table)

        # Store these for use in tests
        cls.cn2_model = cn2_model
        cls.X = X
        cls.y = y
        cls.table = table

    def test_same_number_of_rules(self):
        """
        Checks if the ClassificationRuleSet has the same number of rules
        as the Orange CN2 model (excluding the default rule).
        """
        OrangeCN2FactoryClass = get_orange_cn2_factory_class()
        factory = OrangeCN2FactoryClass()

        ruleset: ClassificationRuleSet = factory.make(self.cn2_model, self.X)

        # In Orange, the last rule is the default rule
        expected_rule_count = len(self.cn2_model.rule_list) - 1
        actual_rule_count = len(ruleset.rules)

        self.assertEqual(
            expected_rule_count,
            actual_rule_count,
            "ClassificationRuleSet should have the same number of rules as CN2 (excluding the default rule)."
        )

    def test_if_prediction_same_as_cn2(self):
        """
        Verifies that the ClassificationRuleSet predictions
        match the original Orange CN2Classifier predictions.
        """
        OrangeCN2FactoryClass = get_orange_cn2_factory_class()
        factory = OrangeCN2FactoryClass()

        ruleset: ClassificationRuleSet = factory.make(self.cn2_model, self.X)

        # Use the FirstRuleCoveringStrategy to mimic CN2 behavior
        ruleset.set_prediction_strategy(FirstRuleCoveringStrategy)

        # Orange's CN2 predict(...) typically returns class probabilities.
        # We use argmax(...) to get the predicted class index.
        cn2_pred = np.argmax(self.cn2_model.predict(self.table.X), axis=1)

        # ClassificationRuleSet predictions
        ruleset_pred = ruleset.predict(self.X)

        self.assertTrue(
            np.array_equal(cn2_pred, ruleset_pred),
            "CN2 factory ruleset predictions should match the original CN2 model."
        )

    def test_serialization(self):
        """
        Tests serialization and deserialization of the ClassificationRuleSet,
        ensuring the predictions remain the same.
        """
        OrangeCN2FactoryClass = get_orange_cn2_factory_class()
        factory = OrangeCN2FactoryClass()

        ruleset: ClassificationRuleSet = factory.make(self.cn2_model, self.X)
        ruleset.set_prediction_strategy(FirstRuleCoveringStrategy)
        original_pred = ruleset.predict(self.X)

        # Serialize -> Deserialize
        serialized = JSONSerializer.serialize(ruleset)
        deserialized: ClassificationRuleSet = JSONSerializer.deserialize(
            serialized, ClassificationRuleSet
        )

        # (Optional) Update coverage/statistics in the deserialized object
        deserialized.update(self.X, self.y)
        deserialized.set_prediction_strategy(FirstRuleCoveringStrategy)
        deserialized_pred = deserialized.predict(self.X)

        self.assertTrue(
            np.array_equal(original_pred, deserialized_pred),
            "Deserialized CN2 ruleset should produce the same predictions as the original ruleset."
        )

    def test_if_subconditions_are_in_the_same_order(self):
        """
        Verifies that subconditions in each rule premise keep the same order
        as in the original Orange CN2 model.
        """
        OrangeCN2FactoryClass = get_orange_cn2_factory_class()
        factory = OrangeCN2FactoryClass()

        ruleset: ClassificationRuleSet = factory.make(self.cn2_model, self.X)

        # In Orange, the last rule is the default rule (no selectors),
        # so only compare up to len(rule_list) - 1
        for rule_index, rule in enumerate(ruleset.rules):
            orange_rule = self.cn2_model.rule_list[rule_index]
            for sub_idx, subcond in enumerate(rule.premise.subconditions):
                # The expected attribute name in decision_rules
                subcond_column_idx = list(subcond.attributes)[0]
                expected_attr_name = ruleset.column_names[subcond_column_idx]

                # The actual attribute name in Orange
                orange_selector = orange_rule.selectors[sub_idx]
                actual_attr_name = orange_selector.variable.name

                self.assertEqual(
                    expected_attr_name,
                    actual_attr_name,
                    "Subconditions should match in order and attribute name."
                )
