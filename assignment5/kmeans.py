import sys

import pandas as pd

from libs import k_means

REQUIRED_ARGC = 3


def main():
    if len(sys.argv) != REQUIRED_ARGC:
        print(f'Usage: {sys.argv[0]} <input_file> <k>')
        return

    arr = pd.read_csv(sys.argv[1], header=None).to_numpy()[:, 2:]
    k_means(arr, int(sys.argv[2]))


if __name__ == '__main__':
    main()
