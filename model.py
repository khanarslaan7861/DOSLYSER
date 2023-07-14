import pandas as pd
import numpy as np

df = pd.read_csv("pre_processed_dataset.csv", low_memory=False)
print(df.describe())
print(df.shape)
df.info()
data = df.to_numpy()

# Data Feature Split
n_samples, n_features = data.shape[0], data.shape[1] - 1
x, y = data[:, 0:n_features], data[:, n_features]
