import pandas as pd


def main():
    df: pd.DataFrame = pd.read_csv('dating-binned.csv')
    test_df = df.sample(frac=0.2, random_state=47)
    train_df = df.drop(test_df.index)  # pylint: disable=maybe-no-member
    test_df.to_csv('testSet.csv', index=False)
    train_df.to_csv('trainSet.csv', index=False)


if __name__ == '__main__':
    main()
