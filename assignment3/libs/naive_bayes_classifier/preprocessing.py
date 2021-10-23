from typing import Dict, Iterable, Set

import pandas as pd

from .definitions import CONTINUOUS_VALUED_COLUMN_INCLUSIVE_RANGE


def encode_label_on_cols(df: pd.DataFrame, cols: Iterable[str]) -> dict[str, dict[str, int]]:
    '''Side effect: `df` is modified in place.'''

    def compile_encoding(series: pd.Series) -> dict:
        return {v: i for i, v in enumerate(sorted(set(series)))}

    encodings = {}

    for col in cols:
        encodings[col] = compile_encoding(df[col])
        df[col] = df[col].map(encodings[col])

    return encodings


def categorize_continuous_columns(df: pd.DataFrame, bin_size: int):
    '''Side effect: `df` is modified in place.'''

    for col, col_range in CONTINUOUS_VALUED_COLUMN_INCLUSIVE_RANGE.items():
        full_range_list = df[col].to_list() + list(col_range)
        binned = pd.cut(full_range_list, bin_size, labels=False)
        removed_max_min = binned[:-len(col_range)]
        df[col] = pd.Series(removed_max_min)


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
