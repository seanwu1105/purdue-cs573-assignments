import matplotlib.pyplot as plt
import pandas as pd

from naive_bayes_classifier import definitions


def main():
    df: pd.DataFrame = pd.read_csv('dating.csv')

    for col in definitions.RATING_OF_PARTNER_FROM_PARTICIPANT:
        rates: pd.Series = df.groupby(col)['decision'].mean()

        plt.scatter(rates.index, rates.array)
        plt.title(col)
        plt.xlabel('Value')
        plt.ylabel('Success Rate (Decision = 1)')
        plt.savefig(f'2_2_{col}.png')
        plt.clf()


if __name__ == '__main__':
    main()
