from typing import List

import numpy as np
import pandas as pd

__version__ = '0.1.0'


# pylint: disable=too-many-arguments
def train(features_list: np.ndarray, output_list: np.ndarray, l2_regulation: float, initial_weights: List[float],
          learning_rate: float, iterations: int, threshold: float) -> list:
    # Insert 1s column to the left of the features matrix to account for the bias (w0)
    features_list = np.insert(features_list, 0, 1, axis=1)

    weights = np.array(initial_weights)
    for features in features_list:
        print(predict(features, weights))
    return []


def predict(features: np.ndarray, weights: np.ndarray) -> float:
    return 1 / (1 + np.exp(-np.dot(features, weights)))


np.set_printoptions(suppress=True)
