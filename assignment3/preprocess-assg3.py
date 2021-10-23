from typing import Dict, List

import pandas as pd

from libs.naive_bayes_classifier.preprocessing import (
    categorize_continuous_columns, encode_label_on_cols)
from libs.preprocessing import (CATEGORICAL_COLS, encode_label,
                                lowercase_on_cols, normalize_preference_scores,
                                split_train_test_sets, strip_quotes_on_cols)


def main():
    raw_data_filename = 'dating-full.csv'
    nrows = 6500
    df = pd.read_csv(raw_data_filename, nrows=nrows)

    cols_need_striped = ('race', 'race_o', 'field')
    strip_quotes_on_cols(df, cols_need_striped)

    cols_need_lowercased = ('field',)
    lowercase_on_cols(df, cols_need_lowercased)

    df = df.apply(normalize_preference_scores, axis=1)

    preprocess_for_lr_svm(df.copy())
    preprocess_for_nbc(df.copy())


def preprocess_for_lr_svm(df: pd.DataFrame):
    encodings: Dict[str, Dict[str, List[int]]] = {}
    for col in CATEGORICAL_COLS:
        encodings[col] = encode_label(df, col)

    print_encoding('gender', 'female', encodings)
    print_encoding('race', 'Black/African American', encodings)
    print_encoding('race_o', 'Other', encodings)
    print_encoding('field', 'economics', encodings)

    split_train_test_sets(df, 'trainingSet.csv', 'testSet.csv')


def print_encoding(col: str, label: str, encodings: Dict[str, Dict[str, List[int]]]):
    print(
        f'Mapped vector for {label} in column {col}: {encodings[col][label]}')


def preprocess_for_nbc(df: pd.DataFrame):
    cols_need_encoded = ('gender', 'race', 'race_o', 'field')
    encode_label_on_cols(df, cols_need_encoded)

    bin_size = 5
    categorize_continuous_columns(df, bin_size)

    split_train_test_sets(df, 'trainingSet_NBC.csv', 'testSet_NBC.csv')


if __name__ == '__main__':
    main()
