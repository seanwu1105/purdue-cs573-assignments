import sys

import pandas as pd

from naive_bayes_classifier.preprocessing import (
    CONTINUOUS_VALUED_COLUMN_INCLUSIVE_RANGE, categorize_continuous_columns,
    count_bins)


def main(input_csv, output_csv):
    df: pd.DataFrame = pd.read_csv(input_csv)
    bin_size = 5

    categorize_continuous_columns(df, bin_size)

    for col in CONTINUOUS_VALUED_COLUMN_INCLUSIVE_RANGE:
        print(f'{col}: {count_bins(df, col, bin_size)}')

    df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python discretize.py <input_csv> <output_csv>')
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
