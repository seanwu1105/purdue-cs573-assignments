from typing import Iterable, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import cv, decision_tree


# pylint: disable=too-many-arguments
def train(data: npt.NDArray[np.integer],
          labels: npt.NDArray[np.integer], size: int,
          max_depth: Optional[int] = None,
          min_data_size_in_leaf: Optional[int] = None,
          attributes_downsampling: bool = False):
    trees = []
    for _ in range(size):
        selected_data, selected_labels = select_bootstrap_sample(data, labels)
        trees.append(
            decision_tree.DecisionTreeClassifier(max_depth,
                                                 min_data_size_in_leaf,
                                                 attributes_downsampling)
            .train(selected_data, selected_labels))

    return trees


def select_bootstrap_sample(data: npt.NDArray[np.integer],
                            labels: npt.NDArray[np.integer]):
    selected_indices = np.random.choice(data.shape[0], size=data.shape[0])
    selected_data = data[selected_indices]
    selected_labels = labels[selected_indices]
    return selected_data, selected_labels


def predict(model: Iterable[decision_tree.Node],
            data: npt.NDArray[np.integer]) -> np.integer:
    predictions = filter(lambda p: p is not None,
                         (decision_tree.predict(tree, data) for tree in model))
    return np.argmax(np.bincount(np.fromiter(predictions, dtype=np.int_)))


def test(data: npt.NDArray[np.integer], labels: npt.NDArray[np.integer],
         model: Iterable[decision_tree.Node]):

    correct = 0
    for features, label in zip(data, labels):
        prediction = predict(model, features)
        if prediction == label:
            correct += 1

    return correct / len(data)


class Rf(cv.Classifier):  # pylint: disable=too-few-public-methods
    class RfModel(cv.Model):  # pylint: disable=too-few-public-methods
        def __init__(self, model: Iterable[decision_tree.Node]):
            self.model = model

        def test(self, data: pd.DataFrame, expect_col: int = -1) -> float:
            np_data = data.to_numpy(dtype=np.int_)
            X = np_data[:, :expect_col]
            y = np_data[:, expect_col]
            return test(X, y, self.model)

    def __init__(self, size: int = 30, max_depth: int = 8,
                 min_data_size_in_leaf: int = 50,
                 attributes_downsampling: bool = False):
        self.size = size
        self.max_depth = max_depth
        self.min_data_size_in_leaf = min_data_size_in_leaf
        self.attributes_downsampling = attributes_downsampling

    def train(self, data: pd.DataFrame, expect_col: int = -1) -> cv.Model:
        np_data = data.to_numpy(dtype=np.int_)
        X = np_data[:, :expect_col]
        y = np_data[:, expect_col]
        return Rf.RfModel(train(X, y, self.size, self.max_depth,
                                self.min_data_size_in_leaf,
                                self.attributes_downsampling))
