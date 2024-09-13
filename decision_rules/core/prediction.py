from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
from decision_rules.core.rule import AbstractConclusion
from decision_rules.core.rule import AbstractRule
from typeguard import typechecked


@typechecked
class PredictionStrategy(ABC):
    """Prediction strategy interface. By subclassing it you can implement
    custom prediction strategies. You have to implement `_perform_prediction`
    method.
    """

    def __init__(
        self,
        rules: list[AbstractRule],
        default_conclusion: AbstractConclusion
    ):
        self.rules: list[AbstractRule] = rules
        self.default_conclusion: AbstractConclusion = default_conclusion
        self.coverage_matrix: np.ndarray

    @abstractmethod
    def _perform_prediction(self, voting_matrix: np.ndarray) -> np.ndarray:
        """Method implementing prediction behavior. You should implement this
        method in your own prediction strategies.

        Args:
            voting_matrix (np.ndarray): special array used for prediction.
                It contains as many rows as there are examples in the dataset
                and as many columns as there are rules in the ruleset. It behaves
                like a coverage matrix but instead of 1 or 0 values it contains rules'
                voting weights if a rule covers given example or 0 value otherwise.

        Returns:
            np.ndarray: predictions
        """

    def predict(self, coverage_matrix: np.ndarray) -> np.ndarray:
        """Predicts for given dataset.

        Args:
            coverage_matrix (np.ndarray): coverage matrix.
                See "AbstractRuleSet.calculate_coverage_matrix" for more details.

        Returns:
            np.ndarray: predictions
        """
        self.coverage_matrix = coverage_matrix
        voting_matrix: np.ndarray = self._calculate_voting_matrix()
        return self._perform_prediction(voting_matrix)

    def _calculate_voting_matrix(self) -> np.ndarray:
        """Prepare special array used for prediction. It contains as many rows as there are
        examples in the dataset and as many columns as there are rules in the ruleset. It
        behaves like a coverage matrix but instead of 1 or 0 values it contains rules' voting weights
        if a rule covers given example or 0 value otherwise.

        Returns:
            np.ndarray: voting matrix
        """
        rules_voting_weights: np.ndarray = np.array([
            rule.voting_weight for rule in self.rules
        ])
        voting_matrix: np.ndarray = self.coverage_matrix * rules_voting_weights

        return voting_matrix


@typechecked
class _PredictionModel(ABC):
    """Prediction model interface. Every ruleset class having the "predict" method should
    implement this interface."""

    def __init__(self) -> None:
        super().__init__()

        self.rules: list[AbstractRule] = None
        self.default_conclusion: AbstractConclusion = None

        self._prediction_strategy_class: Optional[Type[PredictionStrategy]] = None

    def set_prediction_strategy(self, strategy: Union[Type[PredictionStrategy], str]):
        """Sets prediction strategy for this model

        Args:
            strategy (Union[type, str]): either a class implementing
                PredictionStrategy interface or string
                that is a valid key in prediction_strategies_choice dictionary
        """
        if isinstance(strategy, str):
            try:
                self._prediction_strategy_class = self.prediction_strategies_choice[strategy]
            except KeyError as error:
                raise ValueError(
                    f'Unknown prediction strategy: "{strategy}". ' +
                    'Possible values are: ' +
                    f'{list(self.prediction_strategies_choice.keys())}.'
                ) from error
            return
        self._prediction_strategy_class = strategy

    def _get_prediction_strategy(self) -> PredictionStrategy:
        """Returns prediction strategy instance currently used by this model

        Returns:
            PredictionStrategy: class instance inheriting from PredictionStrategy
        """
        if self._prediction_strategy_class is None:
            return self.get_default_prediction_strategy_class()(
                rules=self.rules, default_conclusion=self.default_conclusion
            )
        else:
            return self._prediction_strategy_class(
                rules=self.rules,
                default_conclusion=self.default_conclusion
            )

    @abstractmethod
    def _validate_object_state_before_prediction(self):
        """Method used to validate object state before prediction.

        Raises:
            decision_rules.core.exceptions.InvalidStateError: when object in in a wrong
                state to perform prediction
        """

    @property
    @abstractmethod
    def prediction_strategies_choice(self) -> dict[str, Type[PredictionStrategy]]:
        """Specifies prediction strategies available for this model.

        Returns:
            dict[str, Type[PredictionStrategy]]: Dictionary containing available prediction
                strategies. Keys are prediction strategies names and values are classes
                implementing PredictionStrategy interface for this model.
        """

    @abstractmethod
    def get_default_prediction_strategy_class(self) -> Type[PredictionStrategy]:
        """Returns default prediction strategy class used when user doesn't specify any.

        Returns:
            Type[PredictionStrategy]: class implementing PredictionStrategy
                 interface
        """

    def _map_prediction_values(self, predictions: np.ndarray) -> np.ndarray:
        """This method can be overridden  and used to map prediction values returned
        by prediction strategy before returning it from your model. By default it
        returns predictions passed to it unchanged.

        Args:
            predictions (np.ndarray): raw unmapped predictions

        Returns:
            np.ndarray: mapped predictions
        """
        return predictions

    def predict_using_coverage_matrix(
        self,
        coverage_matrix: np.ndarray,  # pylint: disable=invalid-name
    ) -> np.ndarray:
        """
        Perform prediction using  coverage matrix instead of original
        dataset. Coverage matrix could be calculated using "calculate_coverage_matrix".
        This method could be used to optimize and speed up some calculations as calculating
        rules coverage is an expensive operation.
        Args:
            coverage_matrix (np.ndarray) coverage matrix
        Returns:
            np.ndarray: prediction
        """
        self._validate_object_state_before_prediction()
        strategy: PredictionStrategy = self._get_prediction_strategy()
        predictions: np.ndarray = strategy.predict(coverage_matrix)
        return self._map_prediction_values(predictions)


class BestRulePredictionStrategy(PredictionStrategy):
    """Best rule prediction strategy for prediction.
    """

    def _perform_prediction(self, voting_matrix: np.ndarray) -> np.ndarray:
        not_covered_examples_mask = np.sum(voting_matrix, axis=1) == 0
        best_rules_indices = np.argmax(voting_matrix, axis=1)
        prediction = np.array([
            self._get_prediction_from_conclusion(self.rules[index].conclusion)
            for index in best_rules_indices
        ])
        prediction[not_covered_examples_mask] = self._get_prediction_from_conclusion(
            self.default_conclusion
        )
        return prediction

    def _get_prediction_from_conclusion(self, conclusion: AbstractConclusion) -> Any:
        return conclusion.value
