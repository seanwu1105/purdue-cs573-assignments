import pandas as pd

from naive_bayes_classifier.preprocess import (
    categorize_continuous_columns, continuous_valued_column_inclusive_range,
    count_bins)


def main():
    df: pd.DataFrame = pd.read_csv('dating.csv')
    num_category = 5

    categorize_continuous_columns(df, num_category)

    for col in continuous_valued_column_inclusive_range:
        print(f'{col}: {count_bins(df, col, num_category)}')

    df.to_csv('dating-binned.csv', index=False)


if __name__ == '__main__':
    main()
