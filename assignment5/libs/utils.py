import numpy as np
import numpy.typing as npt


def get_clusters(centroids: npt.NDArray[np.floating],
                 data: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
    return np.fromiter(
        (np.argmin(np.linalg.norm(d - centroids, axis=1)) for d in data),
        dtype=np.int_)
