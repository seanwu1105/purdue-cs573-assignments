'''Run NBC with predefined dataset.'''

from typing import Any, Tuple

import pandas as pd

from . import NaiveBayesClassifier
from .definitions import RANDOM_STATE
from .preprocessing import get_column_sample_spaces

TARGET_NAME = 'decision'


def nbc(t_frac: float, bin_size=5) -> Tuple[float, float]:
    df: pd.DataFrame = pd.read_csv('trainingSet.csv')

    features_name = tuple(k for k in df.keys() if k != TARGET_NAME)

    # pylint: disable=maybe-no-member
    train_df = df.sample(frac=t_frac, random_state=RANDOM_STATE)
    assert isinstance(train_df, pd.DataFrame)

    sample_spaces = get_column_sample_spaces(
        pd.read_csv('dating-binned.csv'), bin_size)
    target_sample_space = sample_spaces[TARGET_NAME]
    feature_sample_spaces = {n: sample_spaces[n] for n in features_name}

    classifier = NaiveBayesClassifier[int](
        train_df, feature_sample_spaces=feature_sample_spaces,
        target_name=TARGET_NAME,
        target_sample_space=target_sample_space)

    test_df = pd.read_csv('testSet.csv')

    return (test(classifier, train_df), test(classifier, test_df))


def test(classifier: NaiveBayesClassifier[Any], test_df: pd.DataFrame):
    count = 0

    def check_result(row: pd.Series):
        nonlocal count
        if classifier.predict(row.drop(TARGET_NAME)) == row[TARGET_NAME]:
            count += 1

    test_df.apply(check_result, axis=1)
    return count / len(test_df)
