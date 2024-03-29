from typing import Union

import numpy as np
import numpy.linalg as la

__version__ = '0.1.0'


# pylint: disable=too-many-arguments
def train(features_list: np.ndarray, expects: np.ndarray,
          initial_weights: np.ndarray,
          l2_regularization: float = 0.01,
          learning_rate: float = 0.001,
          iterations: int = 3500,
          threshold: float = 1e-6) -> np.ndarray:

    # Insert 1s column to the left of the features matrix to account for the
    # bias (w0).
    features_list = np.insert(features_list, 0, 1, axis=1)
    weights = initial_weights.astype(np.float64)

    for _ in range(iterations):

        # Vectorize to calculate gradient decent.
        gradients = np.dot(
            (-expects + logistic(np.dot(features_list, weights))), features_list
        ) / len(features_list)

        regularized_gradients = gradients + l2_regularization * weights

        weights_diff = learning_rate * regularized_gradients
        if la.norm(weights_diff, 2) < threshold:
            break

        weights -= weights_diff

    return weights


def test(features_list: np.ndarray, expects: np.ndarray, model: np.ndarray) -> float:
    features_list = np.insert(features_list, 0, 1, axis=1)

    return np.mean(
        np.where(
            (logistic(np.dot(features_list, model)) > 0.5), 1, 0
        ) == expects
    )


def logistic(x: np.ndarray) -> Union[float, np.ndarray]:
    return 1 / (1 + np.exp(-x))
