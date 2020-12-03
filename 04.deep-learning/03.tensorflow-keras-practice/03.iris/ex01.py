# coding: utf-8
# Iris Species Multi-Class Classification Model
# Explore Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# 데이터 입력
df = pd.read_csv('./dataset/iris.csv', names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

# 데이터 분류
dataset = df.values
x = dataset[:, 0:4].astype(float)
# print(x.shape)

t = dataset[:, 4]
# print(t)

# 문자열을 숫자로 변환
# [1 0 0] = 'Iris-setosa'
# [0 1 0] = 'Iris-versicolor'
# [0 0 1] = 'Iris-virginica'

e = LabelEncoder()
e.fit(t)
t = e.transform(t)
t = tf.keras.utils.to_categorical(t)
print(t)






