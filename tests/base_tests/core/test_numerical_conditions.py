# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import unittest

import numpy as np

from decision_rules import settings
from decision_rules.conditions import ElementaryCondition


class TestNumericalCondition(unittest.TestCase):

    def setUp(self):
        pass

    def test_concise_string_representation(self):
        columns = ['attr']
        condition = ElementaryCondition(
            column_index=0,
            left=1.0,
            right=float('inf'),
            left_closed=True,
            right_closed=False
        )
        self.assertTrue(
            settings.CONCISE_NUMERICAL_CONDITIONS_FORM,
            'Concise string representations for numercial conditions should be enabled by default'
        )
        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = True
        self.assertEqual(condition.to_string(columns), 'attr >= 1.00')
        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = False
        self.assertEqual(condition.to_string(columns), 'attr = <1.00, inf)')

        condition.negated = True

        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = True
        self.assertEqual(condition.to_string(columns), 'attr < 1.00')
        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = False
        self.assertEqual(condition.to_string(columns), 'attr != <1.00, inf)')

        condition.left_closed = False
        condition.negated = False

        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = True
        self.assertEqual(condition.to_string(columns), 'attr > 1.00')
        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = False
        self.assertEqual(condition.to_string(columns), 'attr = (1.00, inf)')

        condition.negated = True

        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = True
        self.assertEqual(condition.to_string(columns), 'attr <= 1.00')
        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = False
        self.assertEqual(condition.to_string(columns), 'attr != (1.00, inf)')

        condition = ElementaryCondition(
            column_index=0,
            left=float('-inf'),
            right=1,
            left_closed=False,
            right_closed=True
        )

        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = True
        self.assertEqual(condition.to_string(columns), 'attr <= 1.00')
        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = False
        self.assertEqual(condition.to_string(columns), 'attr = (-inf, 1.00>')

        condition.negated = True

        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = True
        self.assertEqual(condition.to_string(columns), 'attr > 1.00')
        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = False
        self.assertEqual(condition.to_string(columns), 'attr != (-inf, 1.00>')

        condition.negated = False
        condition.right_closed = False

        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = True
        self.assertEqual(condition.to_string(columns), 'attr < 1.00')
        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = False
        self.assertEqual(condition.to_string(columns), 'attr = (-inf, 1.00)')

        condition.negated = True

        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = True
        self.assertEqual(condition.to_string(columns), 'attr >= 1.00')
        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = False
        self.assertEqual(condition.to_string(columns), 'attr != (-inf, 1.00)')

        settings.CONCISE_NUMERICAL_CONDITIONS_FORM = True

    def test_on_nan_values(self):
        X = np.array([
            np.nan,
            1.0,
        ]).reshape(2, 1)
        condition = ElementaryCondition(
            column_index=0,
            left=float('-inf'),
            right=13.1,
            right_closed=True,
            left_closed=False
        )
        condition.negated = True
        self.assertFalse(
            condition.covered_mask(X)[0],
            'Empty values should not be covered'
        )


if __name__ == '__main__':
    unittest.main()
