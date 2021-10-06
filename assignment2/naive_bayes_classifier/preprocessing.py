from typing import Dict, Iterable, Set

import numpy as np
import pandas as pd

from .definitions import (CONTINUOUS_VALUED_COLUMN_INCLUSIVE_RANGE,
                          PREFERENCE_SCORES_OF_PARTICIPANT,
                          PREFERENCE_SCORES_OF_PARTNER, SAMPLE_FRAC,
                          RANDOM_STATE)


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


def encode_label_on_cols(df: pd.DataFrame, cols: Iterable[str]) -> dict[str, dict[str, int]]:
    '''Side effect: `df` is modified in place.'''

    def compile_encoding(series: pd.Series) -> dict:
        return {v: i for i, v in enumerate(sorted(set(series)))}

    encodings = {}

    for col in cols:
        encodings[col] = compile_encoding(df[col])
        df[col] = df[col].map(encodings[col])

    return encodings


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


def categorize_continuous_columns(df: pd.DataFrame, bin_size: int):
    '''Side effect: `df` is modified in place.'''

    for col, col_range in CONTINUOUS_VALUED_COLUMN_INCLUSIVE_RANGE.items():
        full_range_list = df[col].to_list() + list(col_range)
        binned = pd.cut(full_range_list, bin_size, labels=False)
        removed_max_min = binned[:-len(col_range)]
        df[col] = pd.Series(removed_max_min)


def count_bins(df: pd.DataFrame, col: str, num_category: int) -> np.ndarray:
    return (df.groupby(col)[col]
            .count()
            .reindex(range(num_category), fill_value=0)
            .to_numpy())


def get_categorical_sample_spaces(df: pd.DataFrame) -> Dict[str, Set[int]]:
    categorical_sample_space = {
        'gender': {0, 1},
        'race': None,
        'race_o': None,
        'samerace': {0, 1},
        'field': None,
        'decision': {0, 1}
    }

    for col, values in categorical_sample_space.items():
        if values is None:
            categorical_sample_space[col] = set(df[col].unique())

    return categorical_sample_space  # type: ignore


def get_column_sample_spaces(df: pd.DataFrame, bin_size: int) -> Dict[str, Set[int]]:
    sample_space = get_categorical_sample_spaces(df)

    for col in df.keys():
        if col not in sample_space:
            sample_space[col] = set(range(0, bin_size))

    return sample_space


def split_train_test_sets(input_csv, training_csv, test_csv):
    df: pd.DataFrame = pd.read_csv(input_csv)
    test_df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
    train_df = df.drop(test_df.index)

    assert isinstance(train_df, pd.DataFrame)

    test_df.to_csv(test_csv, index=False)
    train_df.to_csv(training_csv, index=False)
