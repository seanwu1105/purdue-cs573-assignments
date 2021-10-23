import numpy as np
import numpy.linalg as la

__version__ = '0.1.0'


# pylint: disable=too-many-arguments
def train(features_list: np.ndarray, expects: np.ndarray,
          initial_weights: np.ndarray,
          regularization: float = 0.01,
          learning_rate: float = 0.001,
          iterations: int = 2000,
          threshold: float = 1e-6) -> np.ndarray:

    assert np.all((expects == 1) | (expects == -1))

    # Insert 1s column to the left of the features matrix to account for the
    # bias (w0).
    features_list = np.insert(features_list, 0, 1, axis=1)
    weights = initial_weights.astype(np.float64)

    for _ in range(iterations):
        predictions = np.dot(features_list, weights)
        hinge_loss_gradients = (
            features_list.T * np.where(predictions * expects < 1, expects, 0)
        ).T

        gradients = np.sum(regularization * weights - hinge_loss_gradients,
                           axis=0) / len(features_list)

        weights_diff = learning_rate * gradients

        if la.norm(weights_diff, 2) < threshold:
            break

        weights -= weights_diff

    return weights


def test(features_list: np.ndarray, expects: np.ndarray, model: np.ndarray) -> float:
    assert np.all((expects == 1) | (expects == -1))

    features_list = np.insert(features_list, 0, 1, axis=1)

    return np.mean(
        np.where(
            (np.dot(features_list, model) > 0), 1, -1
        ) == expects
    )
