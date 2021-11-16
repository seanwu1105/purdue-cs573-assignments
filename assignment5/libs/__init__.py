__version__ = '0.1.0'

import numpy as np
import numpy.typing as npt


def k_means(data: npt.NDArray[np.float_], k: int, iterations: int = 50):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(iterations):
        clusters = np.fromiter(
            (np.argmin(np.linalg.norm(d - centroids, axis=1)) for d in data),
            dtype=np.int_)

        for i in range(k):
            centroids[i] = np.mean(data[clusters == i], axis=0)

    return centroids
