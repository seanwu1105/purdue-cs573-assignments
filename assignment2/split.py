import sys

import pandas as pd

from naive_bayes_classifier.definitions import RANDOM_STATE, SAMPLE_FRAC


def main(input_csv, training_csv, test_csv):
    df: pd.DataFrame = pd.read_csv(input_csv)
    test_df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
    train_df = df.drop(test_df.index)  # pylint: disable=maybe-no-member
    test_df.to_csv(test_csv, index=False)
    train_df.to_csv(training_csv, index=False)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python split.py <input_csv> <training_csv> <test_csv>')
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
