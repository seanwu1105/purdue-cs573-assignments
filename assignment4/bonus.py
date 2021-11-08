import sys

import pandas as pd

from libs.mlp import Mlp

REQUIRED_ARGC = 3


def main():
    if len(sys.argv) != REQUIRED_ARGC:
        print(
            f'Usage: python {sys.argv[0]} <training_data_filename> ' +
            '<test_data_filename>')
        return

    training_data = pd.read_csv(sys.argv[1])
    test_data = pd.read_csv(sys.argv[2])

    mlp = Mlp()
    model = mlp.train(training_data)
    print(f'Training Accuracy MLP: {model.test(training_data)}')
    print(f'Testing Accuracy MLP: {model.test(test_data)}')


if __name__ == '__main__':
    main()
