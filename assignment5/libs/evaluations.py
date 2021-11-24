import collections

import numpy as np
import numpy.typing as npt
import scipy.spatial
import scipy.stats

from .utils import get_clusters


def sum_of_within_cluster_squared_distances(centroids: npt.NDArray[np.floating],
                                            data: npt.NDArray[np.floating]):
    return sum(np.min(np.linalg.norm(d - centroids, axis=1)**2) for d in data)


def get_silhouette_coefficient(centroids: npt.NDArray[np.floating],
                               data: npt.NDArray[np.floating]):

    clusters = get_clusters(centroids, data)

    intra_distances = []
    for i in range(centroids.shape[0]):
        cluster_data = data[clusters == i]
        if cluster_data.shape[0] <= 1:
            continue
        intra_distances.append(
            np.mean(scipy.spatial.distance.pdist(cluster_data)))

    inter_distances = []
    for i in range(centroids.shape[0]):
        cluster_data = data[clusters == i]
        if cluster_data.shape[0] == 0:
            continue
        cluster_distances = []
        for j in range(centroids.shape[0]):
            if i == j:
                continue
            other_data = data[clusters == j]
            if other_data.shape[0] == 0:
                continue
            cluster_distances.append(np.mean(
                scipy.spatial.distance.cdist(cluster_data, other_data)))
        inter_distances.append(min(cluster_distances))

    return max((inter - intra) / max(intra, inter)
               for intra, inter in zip(intra_distances, inter_distances))


def get_normalized_mutual_information(centroids: npt.NDArray[np.floating],
                                      data: npt.NDArray[np.floating],
                                      labels: npt.NDArray[np.integer]):
    clusters = get_clusters(centroids, data)

    labels_prob = np.bincount(labels).astype(np.float64) / labels.size
    clusters_prob = np.bincount(clusters) / clusters.size

    labels_entropy = scipy.stats.entropy(labels_prob)
    clusters_entropy = scipy.stats.entropy(clusters_prob)

    pairs_counter = collections.Counter(zip(labels, clusters))

    pairs_prob = {k: v / labels.size for k, v in pairs_counter.items()}

    information_gain = sum(prob * np.log(prob / (labels_prob[label] * clusters_prob[cluster]))
                           for (label, cluster), prob in pairs_prob.items())

    return information_gain / (labels_entropy + clusters_entropy)
