import pandas as pd

import lib.preprocessing


def main():
    raw_data_filename = 'dating-full.csv'
    nrows = 6500
    df = pd.read_csv(raw_data_filename, nrows=nrows)
    assert isinstance(df, pd.DataFrame)

    # Drop the columns we don't need.
    cols_need_to_drop = ['race', 'race_o', 'field']
    df.drop(columns=cols_need_to_drop, inplace=True)

    # Encoding gender column.
    lib.preprocessing.encode_label_on_cols(df, ('gender', ))

    # pylint: disable=no-member
    df = df.apply(lib.preprocessing.normalize_preference_scores, axis=1)

    bin_size = 2
    lib.preprocessing.categorize_continuous_columns(df, bin_size)

    lib.preprocessing.split_train_test_sets(
        df, 'trainingSet.csv', 'testSet.csv')


if __name__ == "__main__":
    main()
