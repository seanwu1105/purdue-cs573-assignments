import sys

import pandas as pd

from libs import k_means
from libs.evaluations import (get_normalized_mutual_information,
                              get_silhouette_coefficient,
                              sum_of_within_cluster_squared_distances)

REQUIRED_ARGC = 3


def main():
    if len(sys.argv) != REQUIRED_ARGC:
        print(f'Usage: {sys.argv[0]} <input_file> <k>')
        return

    arr = pd.read_csv(sys.argv[1], header=None).to_numpy()
    data = arr[:, 2:]
    labels = arr[:, 1]
    centroids = k_means(data, int(sys.argv[2]))
    print(
        f'WC-SSD: {sum_of_within_cluster_squared_distances(centroids, data)}')
    print(f'SC: {get_silhouette_coefficient(centroids, data)}')
    print(f'NMI: {get_normalized_mutual_information(centroids, data, labels)}')


if __name__ == '__main__':
    main()
