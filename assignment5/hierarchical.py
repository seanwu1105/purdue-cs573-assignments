from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.cluster

from libs.evaluations import (get_normalized_mutual_information,
                              get_silhouette_coefficient,
                              sum_of_within_cluster_squared_distances)
from libs.utils import get_centroids

EMBEDDING_FILENAME = 'digits-embedding.csv'
DIGITS_SIZE = 10
EACH_DIGIT_SIZE = 10


def main():
    arr = pd.read_csv('digits-embedding.csv', header=None).to_numpy()
    dataset = select_data(arr)
    linkages = plot_cluster_linkages(dataset)
    plot_wc_ssd_and_sc_against_different_k(dataset, linkages)
    get_nmi_with_each_k(dataset, linkages, number_of_clusters=(16, 16, 16))


def select_data(arr: npt.NDArray):
    ret = np.empty((0, arr.shape[1]))
    for i in range(DIGITS_SIZE):
        digits = arr[arr[:, 1] == i]
        selected = digits[np.random.choice(digits.shape[0],
                                           size=EACH_DIGIT_SIZE, replace=False)]
        ret = np.concatenate((ret, selected))
    return ret


def plot_cluster_linkages(dataset: npt.NDArray):
    data = dataset[:, 2:]
    linkages = {
        'Single Linkage': scipy.cluster.hierarchy.single(data),
        'Complete Linkage': scipy.cluster.hierarchy.complete(data),
        'Average Linkage': scipy.cluster.hierarchy.average(data)
    }

    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(15, 5)

    for i, (name, linkage) in enumerate(linkages.items()):
        axes[i].set_title(name)
        scipy.cluster.hierarchy.dendrogram(linkage, ax=axes[i])

    plt.tight_layout()
    plt.show()

    return linkages


def plot_wc_ssd_and_sc_against_different_k(dataset: npt.NDArray,
                                           linkages: dict[str, npt.NDArray]):
    number_of_clusters = (2, 4, 8, 16, 32)
    data = dataset[:, 2:]

    fig, axes = plt.subplots(3, 2)
    fig.set_size_inches(15, 10)

    for idx, (name, linkage) in enumerate(linkages.items()):
        wc_ssd = []
        sc = []
        for k in number_of_clusters:
            clusters = scipy.cluster.hierarchy.fcluster(linkage, k,
                                                        criterion='maxclust') - 1
            centroids = get_centroids(clusters, data)
            wc_ssd.append(
                sum_of_within_cluster_squared_distances(centroids, data))
            sc.append(get_silhouette_coefficient(centroids, data))
        axes[idx][0].plot(number_of_clusters, wc_ssd)
        axes[idx][0].set_title(name)
        axes[idx][0].set_xlabel('Number of Cluster')
        axes[idx][0].set_ylabel('WC SSD')
        axes[idx][1].plot(number_of_clusters, sc)
        axes[idx][1].set_title(name)
        axes[idx][1].set_xlabel('Number of Cluster')
        axes[idx][1].set_ylabel('SC')

    plt.tight_layout()
    plt.show()


def get_nmi_with_each_k(dataset: npt.NDArray, linkages: dict[str, npt.NDArray],
                        number_of_clusters: Iterable[int]):
    data = dataset[:, 2:]
    labels = dataset[:, 1].astype(np.int_)

    for (name, linkage), k in zip(linkages.items(), number_of_clusters):
        clusters = scipy.cluster.hierarchy.fcluster(linkage, k,
                                                    criterion='maxclust') - 1
        centroids = get_centroids(clusters, data)
        print(
            f'{name} NMI: {get_normalized_mutual_information(centroids, data, labels)}')


if __name__ == '__main__':
    main()
