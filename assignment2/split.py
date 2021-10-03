import pandas as pd

from naive_bayes_classifier.definitions import RANDOM_STATE, SAMPLE_FRAC


def main():
    df: pd.DataFrame = pd.read_csv('dating-binned.csv')
    test_df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
    train_df = df.drop(test_df.index)  # pylint: disable=maybe-no-member
    test_df.to_csv('testSet.csv', index=False)
    train_df.to_csv('trainingSet.csv', index=False)


if __name__ == '__main__':
    main()
