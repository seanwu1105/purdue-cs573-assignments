import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

RAW_FILENAME = 'digits-raw.csv'
EMBEDDING_FILENAME = 'digits-embedding.csv'


def draw_random_image():
    df = pd.read_csv(RAW_FILENAME, header=None)
    index = random.randint(0, len(df))
    image_array = df.iloc[index].to_numpy()[2:].reshape(28, 28)
    plt.imshow(image_array, cmap='gray')
    plt.show()


def draw_random_embedding():
    data = pd.read_csv(EMBEDDING_FILENAME, header=None).to_numpy()
    indices = np.random.randint(0, len(data), size=1000)

    plt.scatter(data[indices, 2], data[indices, 3], c=data[indices, 1])
    plt.show()


if __name__ == '__main__':
    draw_random_image()
    draw_random_embedding()
