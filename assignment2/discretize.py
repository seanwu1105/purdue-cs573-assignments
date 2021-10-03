import pandas as pd

from naive_bayes_classifier.preprocessing import (
    CONTINUOUS_VALUED_COLUMN_INCLUSIVE_RANGE, categorize_continuous_columns,
    count_bins)


def main():
    df: pd.DataFrame = pd.read_csv('dating.csv')
    bin_size = 5

    categorize_continuous_columns(df, bin_size)

    for col in CONTINUOUS_VALUED_COLUMN_INCLUSIVE_RANGE:
        print(f'{col}: {count_bins(df, col, bin_size)}')

    df.to_csv('dating-binned.csv', index=False)


if __name__ == '__main__':
    main()
