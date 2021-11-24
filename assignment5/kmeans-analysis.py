import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from libs import k_means
from libs.evaluations import (get_normalized_mutual_information,
                              get_silhouette_coefficient,
                              sum_of_within_cluster_squared_distances)
from libs.utils import get_clusters

EMBEDDING_FILENAME = 'digits-embedding.csv'


def main():
    arr = pd.read_csv(EMBEDDING_FILENAME, header=None).to_numpy()
    dataset_list = (arr,
                    arr[(arr[:, 1] == 2) | (arr[:, 1] == 4) |
                        (arr[:, 1] == 6) | (arr[:, 1] == 7)],
                    arr[(arr[:, 1] == 6) | (arr[:, 1] == 7)])
    if sys.argv[1] == '0':
        plot_different_num_of_clusters(dataset_list)
    elif sys.argv[1] == '1':
        plot_batch_different_num_of_clusters(dataset_list)
    elif sys.argv[1] == '2':
        scatter_clusters(dataset_list, (8, 4, 2))
    else:
        raise ValueError('Invalid argument')


def plot_different_num_of_clusters(dataset_list: Iterable[npt.NDArray]):
    numbers_of_clusters = (2, 4, 8, 16, 32)

    fig, axes = plt.subplots(3, 2)
    fig.set_size_inches(15, 10)

    for idx, dataset in enumerate(dataset_list):
        data = dataset[:, 2:]
        wc_ssd = []
        sc = []
        for number_of_clusters in numbers_of_clusters:
            centroids = k_means(data, number_of_clusters)
            wc_ssd.append(
                sum_of_within_cluster_squared_distances(centroids, data))
            sc.append(get_silhouette_coefficient(centroids, data))
        axes[idx][0].plot(numbers_of_clusters, wc_ssd)
        axes[idx][0].set_title(f'Dataset {idx + 1}')
        axes[idx][0].set_xlabel('Number of Clusters')
        axes[idx][0].set_ylabel('WC SSD')
        axes[idx][1].plot(numbers_of_clusters, sc)
        axes[idx][1].set_title(f'Dataset {idx + 1}')
        axes[idx][1].set_xlabel('Number of Clusters')
        axes[idx][1].set_ylabel('SC')

    plt.tight_layout()
    plt.show()


# pylint: disable=too-many-locals
def plot_batch_different_num_of_clusters(dataset_list: Iterable[npt.NDArray]):
    numbers_of_clusters = (2, 4, 8, 16, 32)
    seeds = range(10)

    fig, axes = plt.subplots(3, 2)
    fig.set_size_inches(15, 10)

    for idx, dataset in enumerate(dataset_list):
        data = dataset[:, 2:]
        wc_ssd_mean = []
        wc_ssd_deviation = []
        sc_mean = []
        sc_deviation = []
        for number_of_clusters in numbers_of_clusters:
            wc_ssd = []
            sc = []
            for seed in seeds:
                np.random.seed(seed)
                centroids = k_means(data, number_of_clusters)
                wc_ssd.append(
                    sum_of_within_cluster_squared_distances(centroids, data))
                sc.append(get_silhouette_coefficient(centroids, data))
            wc_ssd_mean.append(np.mean(wc_ssd))
            wc_ssd_deviation.append(np.std(wc_ssd))
            sc_mean.append(np.mean(sc))
            sc_deviation.append(np.std(sc))
        axes[idx][0].errorbar(numbers_of_clusters, wc_ssd_mean,
                              yerr=wc_ssd_deviation)
        axes[idx][0].set_title(f'Dataset {idx + 1}')
        axes[idx][0].set_xlabel('Number of Clusters')
        axes[idx][0].set_ylabel('WC SSD')
        axes[idx][1].errorbar(numbers_of_clusters, sc_mean,
                              yerr=sc_deviation)
        axes[idx][1].set_title(f'Dataset {idx + 1}')
        axes[idx][1].set_xlabel('Number of Clusters')
        axes[idx][1].set_ylabel('SC')

    plt.tight_layout()
    plt.show()


def scatter_clusters(dataset_list: Iterable[npt.NDArray],
                     numbers_of_cluster: Iterable[int]):
    for idx, (dataset, k) in enumerate(zip(dataset_list, numbers_of_cluster)):
        data = dataset[:, 2:]
        labels = dataset[:, 1].astype(np.int_)
        centroids = k_means(data, k)
        print(
            f'Dataset {idx + 1} NMI: {get_normalized_mutual_information(centroids, data, labels)}')
        indices = np.random.randint(0, len(data), size=1000)
        clusters = get_clusters(centroids, data[indices])
        plt.scatter(data[indices, 0], data[indices, 1], c=clusters, marker='.')
        plt.title(f'Dataset {idx + 1}, k = {k}')
        plt.show()


if __name__ == '__main__':
    main()
