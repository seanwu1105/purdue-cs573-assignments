import sys

import pandas as pd

from libs.decision_tree import Dt
from libs.random_forest import Rf

REQUIRED_ARGC = 4


def main():
    if len(sys.argv) != REQUIRED_ARGC:
        print(
            f'Usage: python {sys.argv[0]} <training_data_filename> ' +
            '<test_data_filename> <model_index: 1 (DT) | 2 (BT) | 3 (RF)>')
        return

    training_data = pd.read_csv(sys.argv[1])
    test_data = pd.read_csv(sys.argv[2])
    model_index = int(sys.argv[3])

    if model_index == 1:
        decisionTree(training_data, test_data)
    elif model_index == 2:
        bagging(training_data, test_data)
    elif model_index == 3:
        randomForest(training_data, test_data)
    else:
        print('Invalid model index')


def decisionTree(training_set: pd.DataFrame, test_set: pd.DataFrame):
    dt = Dt()
    model = dt.train(training_set)
    print(f'Training Accuracy DT: {model.test(training_set):.2f}')
    print(f'Testing Accuracy DT: {model.test(test_set):.2f}')


def bagging(training_set: pd.DataFrame, test_set: pd.DataFrame):
    model = Rf(max_depth=8, min_data_size_in_leaf=50,
               attributes_downsampling=False).train(training_set)
    print(f'Training Accuracy BT: {model.test(training_set):.2f}')
    print(f'Testing Accuracy BT: {model.test(test_set):.2f}')


def randomForest(training_set: pd.DataFrame, test_set: pd.DataFrame):
    model = Rf(max_depth=8, min_data_size_in_leaf=50,
               attributes_downsampling=True).train(training_set)
    print(f'Training Accuracy RF: {model.test(training_set):.2f}')
    print(f'Testing Accuracy RF: {model.test(test_set):.2f}')


if __name__ == '__main__':
    main()
