import sys

import pandas as pd

import libs.decision_tree

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


def decisionTree(training_set: pd.DataFrame, test_set: pd.DataFrame):
    training_data = training_set.to_numpy(dtype=int)
    data = training_data[:, :-1]
    labels = training_data[:, -1]
    model = libs.decision_tree.DecisionTreeClassifier(
        max_depth=8, min_data_size_in_leaf=50).train(data, labels)
    print(
        f'Training Accuracy DT: {libs.decision_tree.test(data, labels, model):.2f}')

    test_data = test_set.to_numpy(dtype=int)
    data = test_data[:, :-1]
    labels = test_data[:, -1]
    print(
        f'Testing Accuracy DT: {libs.decision_tree.test(data, labels, model):.2f}')


if __name__ == '__main__':
    main()
