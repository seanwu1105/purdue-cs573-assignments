import sys

import numpy as np
import pandas as pd

import libs.logistic_regression
import libs.svm
from libs.preprocessing import CATEGORICAL_COLS, convert_to_ndarray

REQUIRED_ARGC = 4


def main():
    if len(sys.argv) != REQUIRED_ARGC:
        print(f'Usage: python {sys.argv[0]} <training_data_filename> ' +
              '<test_data_filename> <output_filename> ' +
              '<model_index: 1 (LR) | 2 (SVM)>')
        return

    training_data: pd.DataFrame = pd.read_csv(sys.argv[1], converters={
        col: convert_to_ndarray for col in CATEGORICAL_COLS})
    test_data: pd.DataFrame = pd.read_csv(sys.argv[2], converters={
        col: convert_to_ndarray for col in CATEGORICAL_COLS})
    model_index = int(sys.argv[3])

    if model_index == 1:
        lr(training_data, test_data)
    elif model_index == 2:
        svm(training_data, test_data)
    else:
        print('Invalid model index')
        return


def lr(training_set: pd.DataFrame, test_set: pd.DataFrame):
    training_data = training_set.to_numpy(dtype=float)
    features_list = training_data[:, :-1]
    expects = training_data[:, -1]
    model = libs.logistic_regression.train(features_list, expects,
                                           initial_weights=np.zeros(
                                               len(training_set.columns)),
                                           l2_regularization=0.01,
                                           learning_rate=0.01,
                                           iterations=500, threshold=1e-6)
    training_accuracy = libs.logistic_regression.test(
        features_list, expects, model)
    print(f'Training Accuracy LR: {training_accuracy:.2f}')

    test_data = test_set.to_numpy(dtype=float)
    features_list = test_data[:, :-1]
    expects = test_data[:, -1]
    test_accuracy = libs.logistic_regression.test(
        features_list, expects, model)
    print(f'Testing Accuracy LR: {test_accuracy:.2f}')


def svm(training_set: pd.DataFrame, test_set: pd.DataFrame):
    training_data = training_set.to_numpy(dtype=float)
    features_list = training_data[:, :-1]
    expects = training_data[:, -1]
    expects[expects == 0] = -1

    model = libs.svm.train(features_list, expects,
                           initial_weights=np.zeros(len(training_set.columns)),
                           regularization=0.01,
                           learning_rate=0.5, iterations=500, threshold=1e-6)

    training_accuracy = libs.svm.test(
        features_list, expects, model)
    print(f'Training Accuracy SVM: {training_accuracy:.2f}')

    test_data = test_set.to_numpy(dtype=float)
    features_list = test_data[:, :-1]
    expects = test_data[:, -1]
    expects[expects == 0] = -1
    test_accuracy = libs.svm.test(
        features_list, expects, model)
    print(f'Testing Accuracy SVM: {test_accuracy:.2f}')


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    main()
