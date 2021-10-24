import abc
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


class Model(abc.ABC):  # pylint: disable=too-few-public-methods
    @abc.abstractmethod
    def test(self, data: pd.DataFrame, expect_col: Union[str, int]) -> float:
        pass


class Classifier(abc.ABC):  # pylint: disable=too-few-public-methods
    @abc.abstractmethod
    def train(self, data: pd.DataFrame, expect_col: Union[str, int]) -> Model:
        pass


class CrossValidation:

    def __init__(self,
                 data: pd.DataFrame,
                 expect_col: Union[str, int],
                 k=10, random_state=18):
        self.expect_col = expect_col
        self.data = partition(data, random_state, k)

    def validate(self,
                 classifier: Classifier,
                 t_frac: float, random_state=32) -> Tuple[Model, float, float]:
        max_accuracy = 0.0
        accuracies = []
        for i, test_df in enumerate(self.data):
            training_df = pd.concat(
                (part for j, part in enumerate(self.data) if j != i))

            training_df = training_df.sample(  # type: ignore
                frac=t_frac, random_state=random_state)
            model = classifier.train(training_df, self.expect_col)
            accuracy = model.test(test_df, self.expect_col)
            accuracies.append(accuracy)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_model = model
        return (best_model,
                np.mean(accuracies),
                np.std(1 - np.array(accuracies)) / np.sqrt(len(accuracies)))


def partition(data: pd.DataFrame, random_state: int, k: int) -> List[pd.DataFrame]:
    data = data.sample(frac=1, random_state=random_state)  # type: ignore
    return np.array_split(data, k)
