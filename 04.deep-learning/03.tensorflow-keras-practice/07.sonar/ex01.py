# coding: utf-8
# Sonar Mineral Binary Classification Model
# Explore Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 데이터 입력
df = pd.read_csv('./dataset/sonar.csv', header=None)

# 데이터 개괄 보기
print(df.info())

# 데이터의 일부분 미리 보기
print(df.head())

# 데이터 분류
dataset = df.values
x = dataset[:, 0:60]
# print(x.shape)

t = dataset[:, 60]
# print(t)

# 문자열을 ONE-HOT로 변환
# [1 0] = 'R'
# [0 1] = 'M'
e = LabelEncoder()
e.fit(t)
t = e.transform(t)
print(t)
