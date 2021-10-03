from typing import Iterable

import numpy as np
import pandas as pd

from .definitions import (continuous_valued_column_inclusive_range,
                          preference_scores_of_participant,
                          preference_scores_of_partner)


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
                                  for col in preference_scores_of_participant)
    for col in preference_scores_of_participant:
        series[col] = series[col] / total_participant_score

    total_partner_score = sum(series[col]
                              for col in preference_scores_of_partner)
    for col in preference_scores_of_partner:
        series[col] = series[col] / total_partner_score
    return series


def categorize_continuous_columns(df: pd.DataFrame, num_category: int):
    '''Side effect: `df` is modified in place.'''

    for col, col_range in continuous_valued_column_inclusive_range.items():
        full_range_list = df[col].to_list() + list(col_range)
        binned = pd.cut(full_range_list, num_category, labels=False)
        removed_max_min = binned[:-len(col_range)]
        df[col] = pd.Series(removed_max_min)


def count_bins(df: pd.DataFrame, col: str, num_category: int) -> np.ndarray:
    return (df.groupby(col)[col]
            .count()
            .reindex(range(num_category), fill_value=0)
            .to_numpy())
