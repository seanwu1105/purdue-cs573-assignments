from collections import defaultdict
from typing import Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import cv


class Perceptron:
    def __init__(self):
        self._inputs: npt.NDArray[np.floating] = np.empty(0, dtype=np.float_)
        self.weights: npt.NDArray[np.floating] = np.empty(0, dtype=np.float_)

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        if self._inputs.size == 0:
            self._inputs = value
            self._inputs = np.append(self._inputs, 1)
        else:
            self._inputs[:-1] = value
        if self.weights.size == 0:
            self.weights = np.random.uniform(-1, 1, len(self.inputs))

    @property
    def output(self):
        return sigmoid(np.dot(self.weights, self.inputs))


def sigmoid(x: npt.NDArray[np.floating]):
    return 1 / (1 + np.exp(-x))


def train(data: npt.NDArray[np.floating], labels: npt.NDArray[np.floating],
          epochs: int, init_learning_rate: float, momentum: float,
          network_shape: tuple[int, ...]):

    dataset = np.column_stack((data, labels))
    unique_labels = np.unique(labels)

    network = tuple((Perceptron(),) * size for size in network_shape + (1,))
    weights_diff: dict[Perceptron, np.floating] = defaultdict(
        lambda: np.float_(0))

    for epoch in range(epochs):
        np.random.shuffle(dataset)
        for row in dataset:
            inputs = row[:-1]
            output = feed_forward(inputs, network)
            deltas = pass_backward(normalize(row[-1], unique_labels),
                                   output, network)
            weights_diff = adjust_weights(
                deltas,
                get_learning_rate(epoch, init_learning_rate),
                weights_diff, momentum)
    return network


def feed_forward(inputs: npt.NDArray[np.floating],
                 network: tuple[tuple[Perceptron, ...], ...]):
    outputs = np.empty(0, dtype=np.float_)
    for idx, neurons in enumerate(network):
        if idx == 0:
            outputs = get_neurons_outputs(neurons, inputs)
        else:
            outputs = get_neurons_outputs(neurons, outputs)

    assert len(outputs) == len(network[-1])

    return outputs[0]


def get_neurons_outputs(neurons: Iterable[Perceptron],
                        data: npt.NDArray[np.floating]):

    for neuron in neurons:
        neuron.inputs = data
    return np.fromiter((neuron.output for neuron in neurons), dtype=np.float_)


def normalize(value: np.floating, unique_labels: npt.NDArray[np.floating]):
    """ Normalize expected output. """
    return (2 * (value - np.amin(unique_labels)) + 1) / (2 * len(unique_labels))


def pass_backward(expected: np.floating,
                  output: npt.NDArray[np.floating],
                  network: tuple[tuple[Perceptron, ...], ...]):
    deltas: dict[Perceptron, np.floating] = {}
    deltas[network[-1][0]] = ((expected - output)
                              * output * (1 - output))

    for neurons_idx, neurons in reversed(tuple(enumerate(network[:-1]))):
        for neuron_idx, neuron in enumerate(neurons):
            # sum of (delta) * (synaptic weight) for each neuron in next layer
            summed = sum(deltas[n] * n.weights[neuron_idx]
                         for n in network[neurons_idx + 1])

            output = neuron.output
            deltas[neuron] = summed * output * (1 - output)  # type: ignore

    return deltas


def get_learning_rate(epoch: int, initial_rate: float):
    return initial_rate / (1 + epoch)


def adjust_weights(deltas: dict[Perceptron, np.floating],
                   learning_rate: float,
                   old_weights_diff: dict[Perceptron, np.floating],
                   momentum: float):
    for neuron in deltas:
        old_weights_diff[neuron] = (
            old_weights_diff[neuron] * momentum +
            learning_rate * deltas[neuron] * neuron.inputs
        )
        neuron.weights += old_weights_diff[neuron]
    return old_weights_diff


def predict(data: npt.NDArray[np.floating],
            model: tuple[tuple[Perceptron, ...], ...]):
    return feed_forward(data, model)


def test(data: npt.NDArray[np.integer], labels: npt.NDArray[np.integer],
         model: tuple[tuple[Perceptron, ...], ...]):
    correct = 0

    for features, label in zip(data, labels):
        prediction = predict(features, model)
        if abs(prediction - label) < 0.5:
            correct += 1

    return correct / len(data)


class Mlp(cv.Classifier):  # pylint: disable=too-few-public-methods
    class MlpModel(cv.Model):  # pylint: disable=too-few-public-methods
        def __init__(self, model: tuple[tuple[Perceptron, ...], ...]):
            self.model = model

        def test(self, data: pd.DataFrame, expect_col: int = -1) -> float:
            np_data = data.to_numpy(dtype=np.float_)
            X = np_data[:, :expect_col]
            y = np_data[:, expect_col]
            return test(X, y, self.model)

    def __init__(self, epochs: int = 100, init_learning_rate: float = 0.8,
                 momentum: float = 0.1,
                 network_shape: tuple[int, ...] = (4, 2)):
        self.epochs = epochs
        self.learning_rate = init_learning_rate
        self.momentum = momentum
        self.network_shape = network_shape

    def train(self, data: pd.DataFrame, expect_col: int = -1) -> cv.Model:
        np_data = data.to_numpy(dtype=np.float_)
        X = np_data[:, :expect_col]
        y = np_data[:, expect_col]
        return Mlp.MlpModel(train(X, y, self.epochs, self.learning_rate,
                                  self.momentum, self.network_shape))
