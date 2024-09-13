# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import types
import unittest.mock
from typing import Type
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
from decision_rules.core.prediction import _PredictionModel
from decision_rules.core.prediction import PredictionStrategy


class MockedPredictionStrategy(PredictionStrategy):
    def _perform_prediction(self, voting_matrix: np.ndarray) -> np.ndarray:
        return np.zeros(0)


class MockedPredictionModel(_PredictionModel):

    def __init__(self) -> None:
        super().__init__()
        self.rules = []
        self.default_conclusion = Mock()

    def _validate_object_state_before_prediction(self) -> None:
        pass

    @property
    def prediction_strategies_choice(self) -> dict[str, Type[PredictionStrategy]]:
        pass

    def get_default_prediction_strategy_class(self) -> Type[PredictionStrategy]:
        return MockedPredictionStrategy


class TestPredictionModel(unittest.TestCase):

    def test_if_validate_object_state_before_prediction_is_called(self):
        mock_instance = MockedPredictionModel()

        with unittest.mock.patch.object(
            MockedPredictionModel,
            '_validate_object_state_before_prediction',
        ) as validate_object_state_before_prediction_mock:
            mock_instance.predict_using_coverage_matrix(np.zeros((0, 0)))
            self.assertEqual(
                validate_object_state_before_prediction_mock.call_count, 1,
                '_validate_object_state_before_prediction should be called before prediction'
            )

    def test_if_map_prediction_values(self):
        mock_instance = MockedPredictionModel()

        with unittest.mock.patch.object(
            MockedPredictionModel,
            '_map_prediction_values',
        ) as map_prediction_values_mock:
            mock_instance.predict_using_coverage_matrix(np.zeros((0, 0)))
            self.assertEqual(
                map_prediction_values_mock.call_count, 1,
                'map_prediction_values_mock should be called after prediction'
            )

    def test_prediction_strategies_choice(self):
        strategies_choice = {
            'existing_strategy': MockedPredictionStrategy
        }

        with (
            unittest.mock.patch.object(
                MockedPredictionModel,
                'prediction_strategies_choice',
                new_callable=unittest.mock.PropertyMock(
                return_value=strategies_choice)
            ),
            unittest.mock.patch.object(
                MockedPredictionStrategy,
                '_perform_prediction',
            ) as strategy_perform_prediction_mock,
        ):
            mocked_model = MockedPredictionModel()
            # test for non-existing strategy
            with self.assertRaises(ValueError, msg='Should fail for non-existing strategy'):
                mocked_model.set_prediction_strategy('non_existing_strategy')

            # test for existing strategy - passed by name
            mocked_model.set_prediction_strategy(
                list(mocked_model.prediction_strategies_choice.keys())[0]
            )
            # test for existing strategy - passed by class
            mocked_model.set_prediction_strategy(
                list(mocked_model.prediction_strategies_choice.values())[0]
            )

            mocked_model.predict_using_coverage_matrix(np.zeros((0, 0)))
            self.assertEqual(
                strategy_perform_prediction_mock.call_count, 1,
                'Strategy predict method should be called'
            )


if __name__ == '__main__':
    unittest.main()
