import collections

import numpy as np
import numpy.typing as npt
import scipy.spatial
import scipy.stats

from .utils import get_cluster


def sum_of_within_cluster_squared_distances(centroids: npt.NDArray[np.floating],
                                            data: npt.NDArray[np.floating]):
    return sum(np.min(np.linalg.norm(d - centroids, axis=1)**2) for d in data)


def get_silhouette_coefficient(centroids: npt.NDArray[np.floating],
                               data: npt.NDArray[np.floating]):

    clusters = get_cluster(centroids, data)

    intra_distance_mean = np.mean(np.fromiter((
        np.mean(scipy.spatial.distance.pdist(data[clusters == i]))
        for i in range(centroids.shape[0])), dtype=np.float_))

    inter_distance_mean = np.mean(np.fromiter(
        (np.mean(scipy.spatial.distance.cdist(data[clusters == i],
                                              data[clusters != i]))
            for i in range(centroids.shape[0])), dtype=np.float_))

    return (inter_distance_mean - intra_distance_mean) / \
        max(intra_distance_mean, inter_distance_mean)


def get_normalized_mutual_information(centroids: npt.NDArray[np.floating],
                                      data: npt.NDArray[np.floating],
                                      labels: npt.NDArray[np.integer]):
    clusters = get_cluster(centroids, data)

    labels_counter = collections.Counter(labels)
    clusters_counter = collections.Counter(clusters)
    pairs_counter = collections.Counter(zip(labels, clusters))

    labels_entropy = scipy.stats.entropy(
        np.fromiter(labels_counter.values(), dtype=np.int_)
    )
    clusters_entropy = scipy.stats.entropy(
        np.fromiter(clusters_counter.values(), dtype=np.int_)
    )

    pairs_against_counter = {k: labels_counter[k[0]] * clusters_counter[k[1]]
                             for k in pairs_counter.keys()}

    information_gain = scipy.stats.entropy(
        np.fromiter(pairs_counter.values(), dtype=np.int_),
        qk=np.fromiter(pairs_against_counter.values(), dtype=np.int_)
    )

    return information_gain / (labels_entropy + clusters_entropy)
