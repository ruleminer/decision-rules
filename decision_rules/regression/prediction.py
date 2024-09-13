import numpy as np
from decision_rules.core.prediction import PredictionStrategy
from decision_rules.regression.rule import RegressionRule


class VotingPredictionStrategy(PredictionStrategy):
    """Voting prediction strategy for regression prediction.
    """

    def _predict_with_rule(
        self,
        prediction_array: np.ndarray,
        rule_index: int,
        rule_covered_mask: np.ndarray
    ):
        """Fills prediction array with results of a single rule prediction.

        Args:
            prediction_array (np.ndarray): array storing rules predictions
            rule_index (int): index of the rule to predict with
            rule_covered_mask (np.ndarray, optional): rule's coverage mask
        """
        rule: RegressionRule = self.rules[rule_index]
        result: float = rule.conclusion.value * rule.voting_weight

        prediction_array[:, rule_index, 0][
            rule_covered_mask
        ] += result
        prediction_array[:, rule_index, 1][
            rule_covered_mask
        ] += rule.voting_weight

    def _transform_prediction_array_into_prediction(self, prediction_array: np.ndarray):
        results_sums: np.ndarray = np.sum(prediction_array[:, :, 0],  axis=1)
        weights_sums: np.ndarray = np.sum(prediction_array[:, :, 1],  axis=1)
        prediction: np.ndarray = np.full(
            shape=(prediction_array.shape[0],),
            fill_value=self.default_conclusion.value
        )
        predict_mask = weights_sums > 0
        prediction[predict_mask] = (
            results_sums[predict_mask] /
            weights_sums[predict_mask]
        )
        return prediction

    def _perform_prediction(self, voting_matrix: np.ndarray) -> np.ndarray:
        prediction_array: np.ndarray = np.full(
            shape=(voting_matrix.shape[0], len(self.rules), 2),
            fill_value=0.0, dtype=float
        )
        for i in range(len(self.rules)):
            rule_covered_mask = self.coverage_matrix[:, i]
            self._predict_with_rule(prediction_array, i, rule_covered_mask)
        return self._transform_prediction_array_into_prediction(
            prediction_array
        )
