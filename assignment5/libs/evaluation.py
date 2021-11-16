import numpy as np
import numpy.typing as npt
import scipy.spatial


def sum_of_within_cluster_distances(centroids: npt.NDArray[np.floating],
                                    data: npt.NDArray[np.floating]):
    return sum(np.min(np.linalg.norm(d - centroids, axis=1)) for d in data)


def get_silhouette_coefficient(centroids: npt.NDArray[np.floating],
                               data: npt.NDArray[np.floating]):

    clusters = np.fromiter(
        (np.argmin(np.linalg.norm(d - centroids, axis=1)) for d in data),
        dtype=np.int_)

    intra_distance_mean = np.mean(np.fromiter((
        np.mean(scipy.spatial.distance.pdist(data[clusters == i]))
        for i in range(centroids.shape[0])), dtype=np.float_))

    inter_distance_mean = np.mean(np.fromiter(
        (np.mean(scipy.spatial.distance.cdist(data[clusters == i],
                                              data[clusters != i]))
            for i in range(centroids.shape[0])), dtype=np.float_))

    return (inter_distance_mean - intra_distance_mean) / \
        max(intra_distance_mean, inter_distance_mean)
