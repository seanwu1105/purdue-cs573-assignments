__version__ = '0.1.0'

import numpy as np
import numpy.typing as npt

from .utils import get_clusters


def k_means(data: npt.NDArray[np.floating], k: int, iterations: int = 50):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(iterations):
        clusters = get_clusters(centroids, data)

        for i in range(k):
            centroids[i] = np.mean(data[clusters == i], axis=0)

    return centroids
