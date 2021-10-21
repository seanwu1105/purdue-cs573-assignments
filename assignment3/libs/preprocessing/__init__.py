from collections import defaultdict
from typing import Dict, Iterable

import numpy as np
import pandas as pd

RANDOM_STATE = 25
SAMPLE_FRAC = 0.2

CATEGORICAL_COLS = ('gender', 'race', 'race_o', 'field')


def strip_quotes_on_cols(df: pd.DataFrame, cols: Iterable[str]) -> int:
    '''Side effect: `df` is modified in place.'''

    count = 0

    def strip_quotes(s: str) -> str:
        nonlocal count
        is_modified = False
        if s[0] == '"' or s[0] == "'":
            s = s[1:]
            is_modified = True
        if s[-1] == '"' or s[-1] == "'":
            s = s[:-1]
            is_modified = True
        if is_modified:
            count += 1
        return s

    cols = ('race', 'race_o', 'field')
    for col in cols:
        df[col] = df[col].map(strip_quotes)

    return count


def lowercase_on_cols(df: pd.DataFrame, cols: Iterable[str]) -> int:
    '''Side effect: `df` is modified in place.'''

    count = 0

    def lowercase(s: str) -> str:
        nonlocal count
        original = s
        s = s.lower()
        if s != original:
            count += 1
        return s

    for col in cols:
        df[col] = df[col].map(lowercase)

    return count


def encode_label(df: pd.DataFrame, col: str) -> Dict[str, np.ndarray]:

    unique_values = sorted(df[col].unique())

    encoding = defaultdict[str, np.ndarray](lambda: np.zeros(
        len(unique_values) - 1, dtype=int))

    for idx, val in enumerate(unique_values):
        if idx == len(unique_values) - 1:
            encoding[val] = encoding[val]
        else:
            encoding[val][idx] = 1

    df[col] = df[col].map(encoding)

    return encoding


def split_train_test_sets(df: pd.DataFrame, training_csv: str, test_csv: str):
    test_df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
    train_df = df.drop(test_df.index)  # type: ignore

    assert isinstance(train_df, pd.DataFrame)

    test_df.to_csv(test_csv, index=False)
    train_df.to_csv(training_csv, index=False)


def convert_to_ndarray(value: str) -> np.ndarray:
    return np.fromstring(value[1:-1], dtype=int, sep=' ')
