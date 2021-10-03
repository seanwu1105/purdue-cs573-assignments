import pandas as pd
from matplotlib import pyplot as plt

from naive_bayes_classifier.preprocessing import categorize_continuous_columns
from naive_bayes_classifier.runner import nbc


def main():
    fractions = (0.01, 0.1, 0.2, 0.5, 0.6, 0.75, 0.9, 1)
    train_accuracies = []
    test_accuracies = []

    for frac in fractions:
        print(f"frac: {frac}")
        df = pd.read_csv('dating.csv')
        categorize_continuous_columns(df, bin_size=5)
        df.to_csv('dating-binned.csv', index=False)

        train_acc, test_acc = nbc(frac)

        print(f'Training Accuracy: {train_acc:.2f}')
        print(f'Testing Accuracy: {test_acc:.2f}')

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    plt.plot(fractions, train_accuracies, fractions, test_accuracies)


if __name__ == '__main__':
    main()
