import random

import pandas as pd
from matplotlib import pyplot as plt

RAW_FILENAME = 'digits-raw.csv'


def draw_random_image():
    df: pd.DataFrame = pd.read_csv(RAW_FILENAME)
    image_index = random.randint(0, len(df))
    image = df.iloc[image_index]
    image_array = image.values.reshape(28, 28)
    plt.imshow(image_array, cmap='gray')
