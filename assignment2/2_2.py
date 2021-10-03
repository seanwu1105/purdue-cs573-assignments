import sys

import matplotlib.pyplot as plt
import pandas as pd

from naive_bayes_classifier import definitions


def main(filename):
    df: pd.DataFrame = pd.read_csv(filename)

    ncols = 3
    fig, axs = plt.subplots(
        len(definitions.RATING_OF_PARTNER_FROM_PARTICIPANT) // ncols,
        ncols,
        figsize=(10, 8),
        sharex=True, sharey=True)

    for i, col in enumerate(definitions.RATING_OF_PARTNER_FROM_PARTICIPANT):
        rates: pd.Series = df.groupby(col)['decision'].mean()

        axs[i // ncols, i % ncols].scatter(rates.index, rates.array)
        axs[i // ncols, i % ncols].set_title(col)
        axs[i // ncols, i % ncols].set_xlabel('Value')
        axs[i // ncols, i % ncols].set_ylabel('Success Rate (Decision = 1)')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python 2_2.py <input_csv>')
        sys.exit(1)

    main(sys.argv[1])
