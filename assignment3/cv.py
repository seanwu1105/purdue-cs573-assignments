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
    expect_col_nbc = 'decision'

    training_set = pd.read_csv('trainingSet.csv')
    test_set = pd.read_csv('testSet.csv')
    expect_col = -1

    t_fracs = (0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
    stats = {
        'nbc': {'test_acc': [], 'std_err': []},
        'lr': {'test_acc': [], 'std_err': []},
        'svm': {'test_acc': [], 'std_err': []}
    }

    for t_frac in t_fracs:
        print('t_frac:', t_frac)
        nbc_validation = libs.CrossValidation(training_set_nbc, expect_col_nbc)
        best_model, avg_acc, std_err = nbc_validation.validate(Nbc(), t_frac)
        test_acc = best_model.test(test_set_nbc, expect_col_nbc)
        stats['nbc']['test_acc'].append(test_acc)
        stats['nbc']['std_err'].append(std_err)
        print(
            '[Naive Bayesian Classifier]',
            'Test Accuracy:', test_acc,
            'CV Average Accuracy:', avg_acc,
            'Standard Error:', std_err
        )

        validation = libs.CrossValidation(training_set, expect_col)
        best_model, avg_acc, std_err = validation.validate(Lr(), t_frac)
        test_acc = best_model.test(test_set, expect_col)
        stats['lr']['test_acc'].append(test_acc)
        stats['lr']['std_err'].append(std_err)
        print(
            '[Logistic Regression]',
            'Test Accuracy:', test_acc,
            'CV Average Accuracy:', avg_acc,
            'Standard Error:', std_err
        )

        best_model, avg_acc, std_err = validation.validate(Svm(), t_frac)
        test_acc = best_model.test(test_set, expect_col)
        stats['svm']['test_acc'].append(test_acc)
        stats['svm']['std_err'].append(std_err)
        print(
            '[SVM]',
            'Test Accuracy:', test_acc,
            'CV Average Accuracy:', avg_acc,
            'Standard Error:', std_err
        )


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
