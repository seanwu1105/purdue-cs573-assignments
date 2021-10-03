from typing import Any, Dict, Set

import numpy as np
import pandas as pd

from naive_bayes_classifier import NaiveBayesClassifier
from naive_bayes_classifier.definitions import RANDOM_STATE
from naive_bayes_classifier.preprocess import get_column_sample_spaces

# df = pd.DataFrame({'A': ['x', 'x', 'y', 'x', 'x'],
#                    'B': [11, 21, 31, 41, 51],
#                    'C': [True, False, True, False, True]})

# feature_sample_spaces: Dict[str, Set] = {
#     'A': {'x', 'y', 'z'},
#     'B': {1, 11, 21, 31, 41, 51}
# }

# target_sample_space = {True, False}

# # df = pd.read_csv('dating-binned.csv')

# # print(get_column_sample_spaces(df))

# nbc = NaiveBayesClassifier[bool](
#     df,
#     feature_sample_spaces=feature_sample_spaces,
#     target_name='C',
#     target_sample_space=target_sample_space)

# result = nbc.predict(pd.Series({'A': 'z', 'B': 1}))

# print(result)


TARGET_NAME = 'decision'


def nbc(t_frac: float):
    df: pd.DataFrame = pd.read_csv('trainingSet.csv')

    features_name = tuple(k for k in df.keys() if k != TARGET_NAME)

    # pylint: disable=maybe-no-member
    train_df = df.sample(frac=t_frac, random_state=RANDOM_STATE)
    assert isinstance(train_df, pd.DataFrame)

    sample_spaces = get_column_sample_spaces(pd.read_csv('dating-binned.csv'))
    target_sample_space = sample_spaces[TARGET_NAME]
    feature_sample_spaces = {n: sample_spaces[n] for n in features_name}

    classifier = NaiveBayesClassifier[int](
        train_df, feature_sample_spaces=feature_sample_spaces,
        target_name=TARGET_NAME,
        target_sample_space=target_sample_space)

    print(f'Training Accuracy: {test(classifier, train_df):.2f}')

    test_df = pd.read_csv('testSet.csv')
    print(f'Testing Accuracy: {test(classifier, test_df):.2f}')


def test(classifier: NaiveBayesClassifier[Any], test_df: pd.DataFrame):
    count = 0

    def check_result(row: pd.Series):
        nonlocal count
        if classifier.predict(row.drop(TARGET_NAME)) == row[TARGET_NAME]:
            count += 1

    test_df.apply(check_result, axis=1)
    return count / len(test_df)


def main():
    nbc(1)


if __name__ == '__main__':
    main()
