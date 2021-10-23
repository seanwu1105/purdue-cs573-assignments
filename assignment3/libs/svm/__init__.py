import numpy as np

__version__ = '0.1.0'


# pylint: disable=too-many-arguments
def train(features_list: np.ndarray, outputs: np.ndarray,
          regularization: float, initial_weights: np.ndarray,
          learning_rate: float, iterations: int,
          threshold: float) -> np.ndarray:

    # Insert 1s column to the left of the features matrix to account for the
    # bias (w0).
    features_list = np.insert(features_list, 0, 1, axis=1)
    weights = initial_weights.astype(np.float64)

    for _ in range(iterations):
        predictions = np.dot(features_list, weights)
        hinge_loss_gradients = (
            features_list.T * np.where(predictions >= 1, 0, outputs)
        ).T

        gradients = np.sum(regularization * weights - hinge_loss_gradients,
                           axis=0) / len(features_list)

        weights_diff = learning_rate * gradients
        if np.max(np.abs(weights_diff)) < threshold:
            break

        weights -= weights_diff

    return weights
