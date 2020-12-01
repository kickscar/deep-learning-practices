import pandas as pd

df = pd.read_csv("./dataset/housing.csv", delim_whitespace=True, header=None)
print(df.info())
print(df.head())
