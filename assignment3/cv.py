from typing import Any, Union

import numpy as np
import pandas as pd

import libs
import libs.logistic_regression
import libs.svm
from libs.naive_bayes_classifier import NaiveBayesClassifier
from libs.naive_bayes_classifier.preprocessing import get_column_sample_spaces


def main():
    training_set_nbc = pd.read_csv('trainingSet_NBC.csv')
    test_set_nbc = pd.read_csv('testSet_NBC.csv')

    training_set = pd.read_csv('trainingSet.csv')
    test_set = pd.read_csv('testSet.csv')

    nbc_validation = libs.CrossValidation(training_set_nbc, 'decision')
    best_nbc = nbc_validation.validate(Nbc(), 1)
    print(best_nbc.test(test_set_nbc, 'decision'))

    validation = libs.CrossValidation(training_set, -1)
    best_lr = validation.validate(Lr(), 1)
    print(best_lr.test(test_set, -1))

    best_svm = validation.validate(Svm(), 1)
    print(best_svm.test(test_set, -1))


class Nbc(libs.Classifier):  # pylint: disable=too-few-public-methods
    class NbcModel(libs.Model):  # pylint: disable=too-few-public-methods
        def __init__(self, classifier: NaiveBayesClassifier[Any]):
            self.classifier = classifier

        def test(self, data: pd.DataFrame, expect_col: Union[str, int]) -> float:
            count = 0

            def check_result(row: pd.Series):
                nonlocal count
                if self.classifier.predict(row.drop(expect_col)) == row[expect_col]:
                    count += 1

            data.apply(check_result, axis=1)
            return count / len(data)

    def train(self, data: pd.DataFrame, expect_col: Union[str, int]) -> libs.Model:
        assert isinstance(expect_col, str)

        features_name = tuple(k for k in data.keys() if k != expect_col)

        bin_size = 5
        sample_spaces = get_column_sample_spaces(data, bin_size)
        target_sample_space = sample_spaces[expect_col]
        feature_sample_spaces = {n: sample_spaces[n] for n in features_name}

        classifier = NaiveBayesClassifier[int](
            data, feature_sample_spaces=feature_sample_spaces,
            target_name=expect_col,
            target_sample_space=target_sample_space)

        return Nbc.NbcModel(classifier)


class Lr(libs.Classifier):  # pylint: disable=too-few-public-methods
    class LrModel(libs.Model):  # pylint: disable=too-few-public-methods
        def __init__(self, model: np.ndarray):
            self.model = model

        def test(self, data: pd.DataFrame,
                 expect_col: Union[str, int]) -> float:
            assert isinstance(expect_col, int)

            training_data = data.to_numpy(dtype=float)
            features_list = training_data[:, :expect_col]
            expects = training_data[:, expect_col]
            return libs.logistic_regression.test(features_list,
                                                 expects, self.model)

    def train(self, data: pd.DataFrame,
              expect_col: Union[str, int]) -> libs.Model:
        assert isinstance(expect_col, int)

        training_data = data.to_numpy(dtype=float)
        features_list = training_data[:, :expect_col]
        expects = training_data[:, expect_col]
        return Lr.LrModel(
            libs.logistic_regression.train(features_list, expects,
                                           initial_weights=np.zeros(
                                               len(data.columns)),
                                           l2_regularization=0.01,
                                           learning_rate=0.01,
                                           iterations=500, threshold=1e-6))


class Svm(libs.Classifier):  # pylint: disable=too-few-public-methods
    class SvmModel(libs.Model):  # pylint: disable=too-few-public-methods
        def __init__(self, model: np.ndarray):
            self.model = model

        def test(self, data: pd.DataFrame,
                 expect_col: Union[str, int]) -> float:
            assert isinstance(expect_col, int)

            training_data = data.to_numpy(dtype=float)
            features_list = training_data[:, :expect_col]
            expects = training_data[:, expect_col]
            expects[expects == 0] = -1
            return libs.svm.test(features_list,
                                 expects, self.model)

    def train(self, data: pd.DataFrame,
              expect_col: Union[str, int]) -> libs.Model:
        assert isinstance(expect_col, int)

        training_data = data.to_numpy(dtype=float)
        features_list = training_data[:, :expect_col]
        expects = training_data[:, expect_col]
        expects[expects == 0] = -1
        return Svm.SvmModel(
            libs.svm.train(features_list, expects,
                           initial_weights=np.zeros(len(data.columns)),
                           regularization=0.01,
                           learning_rate=0.5, iterations=500, threshold=1e-6))


if __name__ == '__main__':
    main()
