import pandas as pd

from naive_bayes_classifier.preprocess import (
    CONTINUOUS_VALUED_COLUMN_INCLUSIVE_RANGE, categorize_continuous_columns,
    count_bins)


def main():
    df: pd.DataFrame = pd.read_csv('dating.csv')
    num_category = 5

    categorize_continuous_columns(df, num_category)

    for col in CONTINUOUS_VALUED_COLUMN_INCLUSIVE_RANGE:
        print(f'{col}: {count_bins(df, col, num_category)}')

    df.to_csv('dating-binned.csv', index=False)


if __name__ == '__main__':
    main()
