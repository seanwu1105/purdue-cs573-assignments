import sys

from naive_bayes_classifier.preprocessing import split_train_test_sets


def main(input_csv, training_csv, test_csv):
    split_train_test_sets(input_csv, training_csv, test_csv)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python split.py <input_csv> <training_csv> <test_csv>')
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
