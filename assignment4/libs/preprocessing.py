from typing import Iterable

import pandas as pd

from .definitions import (CONTINUOUS_VALUED_COLUMN_INCLUSIVE_RANGE,
                          PREFERENCE_SCORES_OF_PARTICIPANT,
                          PREFERENCE_SCORES_OF_PARTNER, RANDOM_STATE,
                          SAMPLE_FRAC)


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


def split_train_test_sets(df: pd.DataFrame, training_csv: str, test_csv: str):
    test_df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)

    # pylint: disable=maybe-no-member
    train_df = df.drop(test_df.index)  # type: ignore

    assert isinstance(train_df, pd.DataFrame)

    test_df.to_csv(test_csv, index=False)
    train_df.to_csv(training_csv, index=False)
