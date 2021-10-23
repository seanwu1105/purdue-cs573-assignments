import sys

import pandas as pd

import libs.logistic_regression
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
        # from libs.lr_svm import SVM
        # model = SVM()
        pass
    else:
        print('Invalid model index')
        return


def lr(training_set: pd.DataFrame, test_set: pd.DataFrame):
    training_data = training_set.to_numpy(dtype=float)
    features_list = training_data[:, :-1]
    output_list = training_data[:, -1]
    model = libs.logistic_regression.train(features_list, output_list,
                                           l2_regulation=0.01,
                                           initial_weights=[0] *
                                           len(training_set.columns),
                                           learning_rate=0.01,
                                           iterations=500, threshold=1e-6)
    training_accuracy = libs.logistic_regression.test(
        features_list, output_list, model)
    print(f'Training Accuracy LR: {training_accuracy:.2f}')

    test_data = test_set.to_numpy(dtype=float)
    features_list = test_data[:, :-1]
    output_list = test_data[:, -1]
    test_accuracy = libs.logistic_regression.test(
        features_list, output_list, model)
    print(f'Testing Accuracy LR: {test_accuracy:.2f}')


if __name__ == '__main__':
    main()
