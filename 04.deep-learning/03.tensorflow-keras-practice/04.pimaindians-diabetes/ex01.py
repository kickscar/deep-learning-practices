# coding: utf-8
# Pima Indians Diabets Prediction Model
# Explore Dataset
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Pima Indians Diabets Dataset
df = pd.read_csv(
    './dataset/pimaindians-diabetes.csv',
    names=["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])

# 처음 5줄
print(df.head(5))

# 데이터의 전반적인 정보
print(df.info())

# 각 정보별 특징
print(df.describe())

# 데이터 중 임신 정보와 클래스
print(df[['pregnant', 'class']])

# 데이터 간의 상관관계
# colormap = plt.cm.gist_heat   #그래프의 색상 구성
# plt.figure(figsize=(12, 12))   #그래프의 크기
# 그래프의 속성을 결정(vmax의 값을 0.5로 지정해 0.5에 가까울 수록 밝은 색으로 표시)
# sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=colormap, linecolor='white', annot=True)
# plt.show()

grid = sns.FacetGrid(df, col='class')
grid.map(plt.hist, 'plasma',  bins=10)
plt.show()




