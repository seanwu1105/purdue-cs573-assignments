from typing import List, Union

import numpy as np

__version__ = '0.1.0'


# pylint: disable=too-many-arguments
def train(features_list: np.ndarray, outputs: np.ndarray,
          l2_regulation: float, initial_weights: List[float],
          learning_rate: float, iterations: int,
          threshold: float) -> np.ndarray:
    # Insert 1s column to the left of the features matrix to account for the
    # bias (w0).
    features_list = np.insert(features_list, 0, 1, axis=1)
    weights = np.array(initial_weights, dtype=np.float64)

    for _ in range(iterations):
        gradient = np.dot(
            (-outputs + logistic(np.dot(features_list, weights))), features_list
        ) / len(features_list)

        regularized_gradient = gradient + l2_regulation * weights

        weight_diff = learning_rate * regularized_gradient

        if max(abs(weight_diff)) < threshold:
            break

        weights -= weight_diff

    return weights


def test(features_list: np.ndarray, outputs: np.ndarray, model: np.ndarray) -> float:
    features_list = np.insert(features_list, 0, 1, axis=1)

    return np.mean(
        np.where(
            (logistic(np.dot(features_list, model)) > 0.5), 1, 0
        ) == outputs
    )


def logistic(x: np.ndarray) -> Union[float, np.ndarray]:
    return 1 / (1 + np.exp(-x))


np.set_printoptions(suppress=True)
