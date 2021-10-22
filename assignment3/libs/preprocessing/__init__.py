from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

RANDOM_STATE = 25
SAMPLE_FRAC = 0.2

CATEGORICAL_COLS = ('gender', 'race', 'race_o', 'field')

PREFERENCE_SCORES_OF_PARTICIPANT = (
    'attractive_important', 'sincere_important', 'intelligence_important',
    'funny_important', 'ambition_important', 'shared_interests_important')

PREFERENCE_SCORES_OF_PARTNER = (
    'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',
    'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests')


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


def normalize_preference_scores(series: pd.Series) -> pd.Series:
    total_participant_score = sum(series[col]
                                  for col in PREFERENCE_SCORES_OF_PARTICIPANT)
    for col in PREFERENCE_SCORES_OF_PARTICIPANT:
        series[col] = series[col] / total_participant_score

    total_partner_score = sum(series[col]
                              for col in PREFERENCE_SCORES_OF_PARTNER)
    for col in PREFERENCE_SCORES_OF_PARTNER:
        series[col] = series[col] / total_partner_score
    return series


def encode_label(df: pd.DataFrame, col: str) -> Dict[str, List[int]]:

    unique_values = sorted(df[col].unique())

    encoding = defaultdict[str, List[int]](
        lambda: [0] * (len(unique_values) - 1))

    for idx, val in enumerate(unique_values):
        if idx == len(unique_values) - 1:
            encoding[val] = encoding[val]
        else:
            encoding[val][idx] = 1

    # Create new columns to store encoded vectors
    # Get names for new created columns
    names = tuple(f'{col}_{idx}' for idx in range(len(unique_values) - 1))

    col_index = df.columns.get_loc(col)

    # Convert all encodings to 2D array
    encoded = np.array(df[col].map(encoding).to_list())

    for idx, name in enumerate(names):
        df.insert(col_index + idx + 1, name, encoded[:, idx])

    # Remove original column
    df.drop(columns=[col], inplace=True)

    return encoding


def split_train_test_sets(df: pd.DataFrame, training_csv: str, test_csv: str):
    test_df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
    train_df = df.drop(test_df.index)  # type: ignore

    assert isinstance(train_df, pd.DataFrame)

    test_df.to_csv(test_csv, index=False)
    train_df.to_csv(training_csv, index=False)


def convert_to_ndarray(value: str) -> np.ndarray:
    return np.fromstring(value[1:-1], dtype=int, sep=' ')
