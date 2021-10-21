import numpy as np
import pandas as pd

from libs.preprocessing import CATEGORICAL_COLS, convert_to_ndarray

df = pd.read_csv('testSet.csv', converters={
                 col: convert_to_ndarray for col in CATEGORICAL_COLS})

print(type(df['race'][0]))
