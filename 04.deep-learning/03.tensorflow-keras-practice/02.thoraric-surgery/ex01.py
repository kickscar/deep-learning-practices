# coding: utf-8
# Housing Price Prediction(Linear Regression) Model
# Explore Dataset
import pandas as pd

df = pd.read_csv("./dataset/thoraric-surgery.csv", delim_whitespace=True, header=None)
print(df.info())
print(df.head())
