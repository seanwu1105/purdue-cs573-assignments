import sys

import pandas as pd

from libs import k_means
from libs.evaluation import (get_silhouette_coefficient,
                             sum_of_within_cluster_distances)

REQUIRED_ARGC = 3


def main():
    if len(sys.argv) != REQUIRED_ARGC:
        print(f'Usage: {sys.argv[0]} <input_file> <k>')
        return

    arr = pd.read_csv(sys.argv[1], header=None).to_numpy()[:, 2:]
    centroids = k_means(arr, int(sys.argv[2]))
    print(f'WC-SSD: {sum_of_within_cluster_distances(centroids, arr)}')
    print(f'SC: {get_silhouette_coefficient(centroids, arr)}')


if __name__ == '__main__':
    main()
