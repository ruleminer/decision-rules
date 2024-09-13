import numpy as np
from decision_rules.core.prediction import PredictionStrategy


class VotingPredictionStrategy(PredictionStrategy):
    """Voting prediction strategy for classification prediction.
    """

    def _perform_prediction(self, voting_matrix: np.ndarray) -> np.ndarray:
        conclusions = np.array([r.conclusion.value for r in self.rules])
        unique_conclusions = np.unique(conclusions)
        conclusions_values_to_indices_map: dict[str, int] = {
            value: i for i, value in enumerate(unique_conclusions)
        }
        conclusions_indices_to_values_map: dict[str, int] = {
            i: value for i, value in enumerate(unique_conclusions)
        }
        num_unique_conclusions = np.unique(conclusions).shape[0]
        prediction_array_shape = (
            voting_matrix.shape[0],
            num_unique_conclusions,
        )
        prediction_array = np.full(
            prediction_array_shape, fill_value=0.0, dtype=float
        )
        for i, rule in enumerate(self.rules):
            conclusion_index: int = conclusions_values_to_indices_map[rule.conclusion.value]
            prediction_array[:, conclusion_index] += voting_matrix[:, i]

        prediction = np.argmax(prediction_array, axis=1)
        # map prediction values indices for real values
        prediction = np.vectorize(
            conclusions_indices_to_values_map.get)(prediction)
        # predict uncovered examples with default conclusion
        not_covered_examples_mask = np.all(
            np.isclose(prediction_array, 0.0), axis=1
        )
        prediction[not_covered_examples_mask] = self.default_conclusion.value
        return prediction
