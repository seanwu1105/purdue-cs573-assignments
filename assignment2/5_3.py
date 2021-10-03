import pandas as pd

from naive_bayes_classifier.preprocessing import categorize_continuous_columns
from naive_bayes_classifier.runner import nbc


def main():
    fracs = (0.01, 0.1, 0.2, 0.5, 0.6, 0.75, 0.9, 1)

    for frac in fracs:
        print(f"frac: {frac}")
        df = pd.read_csv('dating.csv')
        categorize_continuous_columns(df, bin_size=5)
        df.to_csv('dating-binned.csv', index=False)

        nbc(frac)


if __name__ == '__main__':
    main()
