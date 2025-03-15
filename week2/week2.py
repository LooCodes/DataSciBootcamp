

import numpy as np



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

A = np.array([1,2,3,5,6])
B = np.array([4,5,6,1,2])

# 1.
horizontal = np.hstack((A,B))
vertical = np.vstack((A,B))

# 2.
comm_elemts = np.intersect1d(A,B)

# 3. 
filtered = np.where((A >= 5) & (A <= 10))

# 4.
filtered_rows = iris_2d[(iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)]



import pandas as pd


# 1. 
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
filtered_df = df.loc[::20, ['Manufacturer', 'Model', 'Type']]
print("\nFiltered Rows (Every 20th Row):\n", filtered_df)

# 2.
df['Min.Price'].fillna(df['Min.Price'].mean(), inplace=True)
df['Max.Price'].fillna(df['Max.Price'].mean(), inplace=True)
print("\nUpdated Cars93 Dataset with Filled Missing Values:\n", df[['Min.Price', 'Max.Price']].head())

# 3. 
random_df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
filtered_rows_df = random_df[random_df.sum(axis=1) > 100]
print("\nRows with Sum > 100:\n", filtered_rows_df)