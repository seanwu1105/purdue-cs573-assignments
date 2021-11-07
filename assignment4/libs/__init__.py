import abc

import pandas as pd

__version__ = '0.1.0'


class Model(abc.ABC):  # pylint: disable=too-few-public-methods
    @abc.abstractmethod
    def test(self, data: pd.DataFrame, expect_col: int = -1) -> float:
        pass


class Classifier(abc.ABC):  # pylint: disable=too-few-public-methods
    @abc.abstractmethod
    def train(self, data: pd.DataFrame, expect_col: int = -1) -> Model:
        pass
