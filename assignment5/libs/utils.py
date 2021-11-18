import numpy as np
import numpy.typing as npt


def get_clusters(centroids: npt.NDArray[np.floating],
                 data: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
    return np.fromiter(
        (np.argmin(np.linalg.norm(d - centroids, axis=1)) for d in data),
        dtype=np.int_)


def get_centroids(clusters: npt.NDArray[np.integer],
                  data: npt.NDArray[np.floating]):
    k = len(np.unique(clusters))
    return np.array(tuple(np.mean(data[clusters == i], axis=0)
                          for i in range(k)))
