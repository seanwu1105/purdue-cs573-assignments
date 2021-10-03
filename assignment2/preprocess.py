from typing import Dict

import pandas as pd

from naive_bayes_classifier.preprocess import (
    PREFERENCE_SCORES_OF_PARTICIPANT, PREFERENCE_SCORES_OF_PARTNER,
    encode_label_on_cols, lowercase_on_cols, normalize_preference_scores,
    strip_quotes_on_cols)


def preprocess():
    raw_filename = 'dating-full.csv'
    df = pd.read_csv(raw_filename)

    cols_need_striped = ('race', 'race_o', 'field')
    count = strip_quotes_on_cols(df, cols_need_striped)
    print(f'Quotes removed from {count} cells.')

    cols_need_lowercased = ('field',)
    count = lowercase_on_cols(df, cols_need_lowercased)
    print(f'Standardized {count} cells to lower case.')

    cols_need_encoded = ('gender', 'race', 'race_o', 'field')
    encodings = encode_label_on_cols(df, cols_need_encoded)
    print_encoding('gender', 'male', encodings)
    print_encoding('race', 'European/Caucasian-American', encodings)
    print_encoding('race_o', 'Latino/Hispanic American', encodings)
    print_encoding('field', 'law', encodings)

    df = df.apply(normalize_preference_scores, axis=1)
    for col in PREFERENCE_SCORES_OF_PARTICIPANT:
        print_mean(df, col)
    for col in PREFERENCE_SCORES_OF_PARTNER:
        print_mean(df, col)

    df.to_csv('dating.csv', index=False)


def print_encoding(col: str, label: str, encodings: Dict[str, Dict[str, int]]):
    print(
        f'Value assigned for {label} in column {col}: {encodings[col][label]}.')


def print_mean(df: pd.DataFrame, col: str):
    print(f'Mean of {col}: {df[col].mean():.2f}.')


if __name__ == '__main__':
    preprocess()
