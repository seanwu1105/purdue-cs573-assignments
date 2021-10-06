import pandas as pd
from matplotlib import pyplot as plt

from naive_bayes_classifier.preprocessing import (categorize_continuous_columns,
                                                  split_train_test_sets)
from naive_bayes_classifier.runner import nbc


def main():
    fractions = (0.01, 0.1, 0.2, 0.5, 0.6, 0.75, 0.9, 1)
    train_accuracies = []
    test_accuracies = []

    for frac in fractions:
        print(f"frac: {frac}")

        # Bin the continuous columns
        df = pd.read_csv('dating.csv')
        categorize_continuous_columns(df, bin_size=5)
        df.to_csv('dating-binned.csv', index=False)

        # Split data into training and test sets
        split_train_test_sets('dating-binned.csv',
                              'trainingSet.csv', 'testSet.csv')

        train_acc, test_acc = nbc(frac)

        print(f'Training Accuracy: {train_acc:.2f}')
        print(f'Testing Accuracy: {test_acc:.2f}')

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    _, ax = plt.subplots()
    ax.plot(fractions, train_accuracies, label='Training Accuracy')
    ax.plot(fractions, test_accuracies, label='Testing Accuracy')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
