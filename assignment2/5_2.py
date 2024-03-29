import pandas as pd
from matplotlib import pyplot as plt

from naive_bayes_classifier.preprocessing import (categorize_continuous_columns,
                                                  split_train_test_sets)
from naive_bayes_classifier.runner import nbc


def main():
    bin_sizes = (2, 5, 10, 50, 100, 200)
    train_accuracies = []
    test_accuracies = []

    for bin_size in bin_sizes:
        print(f"Bin size: {bin_size}")

        # Bin continuous columns
        df = pd.read_csv('dating.csv')
        categorize_continuous_columns(df, bin_size)
        df.to_csv('dating-binned.csv', index=False)

        # Split data into training and test sets
        split_train_test_sets('dating-binned.csv',
                              'trainingSet.csv', 'testSet.csv')

        train_acc, test_acc = nbc(1, bin_size)

        print(f'Training Accuracy: {train_acc:.2f}')
        print(f'Testing Accuracy: {test_acc:.2f}')

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    _, ax = plt.subplots()
    ax.plot(bin_sizes, train_accuracies, label='Training Accuracy')
    ax.plot(bin_sizes, test_accuracies, label='Testing Accuracy')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
