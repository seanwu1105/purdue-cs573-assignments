import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

RAW_FILENAME = 'digits-raw.csv'
EMBEDDING_FILENAME = 'digits-embedding.csv'


def draw_random_image():
    arr = pd.read_csv(RAW_FILENAME, header=None).to_numpy()

    images = arr[:, 2:]
    labels = arr[:, 1]
    _, ax = plt.subplots(2, 5)
    for i in range(10):
        indices = (labels == i).nonzero()[0]
        index = np.random.choice(indices)
        ax[i // 5, i % 5].imshow(images[index].reshape(28, 28), cmap='gray')
        ax[i // 5, i % 5].axis('off')

    plt.show()


def draw_random_embedding():
    data = pd.read_csv(EMBEDDING_FILENAME, header=None).to_numpy()
    indices = np.random.randint(0, len(data), size=1000)

    plt.scatter(data[indices, 2], data[indices, 3],
                c=data[indices, 1], marker='.')
    plt.show()


if __name__ == '__main__':
    draw_random_image()
    draw_random_embedding()
