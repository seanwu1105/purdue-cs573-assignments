from typing import Iterable

import pandas as pd


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


preference_scores_of_participant = ([
    'attractive_important', 'sincere_important', 'intelligence_important',
    'funny_important', 'ambition_important', 'shared_interests_important'])

preference_scores_of_partner = ([
    'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',
    'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests'])
