import pandas as pd

import libs.preprocessing


def main():
    raw_data_filename = 'dating-full.csv'
    nrows = 6500
    df = pd.read_csv(raw_data_filename, nrows=nrows)
    assert isinstance(df, pd.DataFrame)

    # Drop the columns we don't need.
    cols_need_to_drop = ['race', 'race_o', 'field']
    df.drop(columns=cols_need_to_drop, inplace=True)

    # Encoding gender column.
    libs.preprocessing.encode_label_on_cols(df, ('gender', ))

    # pylint: disable=no-member
    df = df.apply(libs.preprocessing.normalize_preference_scores, axis=1)

    bin_size = 2
    libs.preprocessing.categorize_continuous_columns(df, bin_size)

    libs.preprocessing.split_train_test_sets(
        df, 'trainingSet.csv', 'testSet.csv')


if __name__ == "__main__":
    main()
