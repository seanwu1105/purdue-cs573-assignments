from typing import Optional

import numpy as np
import numpy.typing as npt


# pylint: disable=too-few-public-methods
class Node:
    def __init__(self, label: Optional[np.integer],
                 attribute: Optional[np.integer],
                 children: dict[np.integer, 'Node']):
        self.label = label
        self.attribute = attribute
        self.children = children


class DecisionTreeClassifier:
    def __init__(self, max_depth: Optional[int] = None,
                 min_data_size_in_leaf: Optional[int] = None):
        self.max_depth = max_depth
        self.min_data_size_in_leaf = min_data_size_in_leaf

    def train(self, data: npt.NDArray[np.integer],
              labels: npt.NDArray[np.integer]):
        return self._build_tree(data, labels, np.arange(data.shape[1]))

    def _build_tree(self, data: npt.NDArray[np.integer],
                    labels: npt.NDArray[np.integer],
                    attributes: npt.NDArray[np.integer],
                    depth: int = 0):
        if data.size == 0:
            raise Exception(
                f'data is empty. labels: {labels}, attributes: {attributes}')

        if np.unique(labels).size == 1:
            return Node(labels[0], None, {})

        if (attributes.size == 0 or depth == self.max_depth
                or self.min_data_size_in_leaf and data.size < self.min_data_size_in_leaf):
            return Node(np.argmax(np.bincount(labels)), None, {})

        best_attribute = choose_best_attribute(data, labels, attributes)

        values = np.unique(data[:, best_attribute])
        children: dict[np.integer, Node] = {}
        for v in values:
            data_v_indices = np.where(data[:, best_attribute] == v)[0]
            child_labels = labels[data_v_indices]
            child_attributes = attributes[attributes != best_attribute]
            children[v] = self._build_tree(data[data_v_indices], child_labels,
                                           child_attributes, depth + 1)

        return Node(None, best_attribute, children)


def choose_best_attribute(data: npt.NDArray[np.integer],
                          labels: npt.NDArray[np.integer],
                          attributes: npt.NDArray[np.integer]):
    best_attribute: np.integer = attributes[0]
    best_gain = 0

    attribute: np.integer
    for attribute in attributes:
        gain = gini_gain(data, labels, attribute)
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute
    return best_attribute


def gini_gain(data: npt.NDArray[np.integer], labels: npt.NDArray[np.integer],
              attribute: np.integer):
    values = np.unique(data[:, attribute])

    new_gini = 0.0
    for v in values:
        data_v_indices = np.where(data[:, attribute] == v)[0]
        new_gini += (len(data_v_indices) / len(data)
                     * gini(labels[data_v_indices]))

    return gini(labels) - new_gini


def gini(labels: npt.NDArray[np.integer]) -> float:
    return 1 - np.sum(np.square(np.bincount(labels) / labels.size))


def predict(model: Node, data: npt.NDArray[np.integer]) -> np.integer:
    if model.label is not None:
        return model.label
    return predict(model.children[data[model.attribute]], data)


def test(data: npt.NDArray[np.integer], labels: npt.NDArray[np.integer],
         model: Node):

    correct = 0
    for features, label in zip(data, labels):
        prediction = predict(model, features)
        if prediction == label:
            correct += 1

    return correct / len(data)
