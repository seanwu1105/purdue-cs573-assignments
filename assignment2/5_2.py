import pandas as pd

from naive_bayes_classifier.preprocess import categorize_continuous_columns
from naive_bayes_classifier.runner import nbc


def main():
    bin_sizes = (2, 5, 10, 50, 100, 200)

    for bin_size in bin_sizes:
        print(f"Bin size: {bin_size}")
        df = pd.read_csv('dating.csv')
        categorize_continuous_columns(df, bin_size)
        df.to_csv('dating-binned.csv', index=False)

        nbc(1, bin_size)


if __name__ == '__main__':
    main()
