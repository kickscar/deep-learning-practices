# coding: utf-8
# Wine Binary Classification Model(와인 종류 분류 모델)
# Model Fitting #1
from keras.models import Sequential
from keras.layers.core import Dense
import pandas as pd
import numpy as np

# 1. load training/test data
df = pd.read_csv('./dataset/wine.csv', header=None)

dataset = df.values
x = dataset[:, 0:12]
t = dataset[:, 12]
t = t[:, np.newaxis]
t = np.c_[t, t == 0]

# 2. model frame config
model = Sequential()
model.add(Dense(30, input_dim=x.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))

# 3. model fitting environment
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. model fitting
model.fit(x, t, epochs=200, batch_size=100, verbose=1)

# 5. result
result = model.evaluate(x, t, verbose=0)
print(f'\n(Loss, Accuracy) = ({result[0]}, {result[1]})')

# 6. predict
# data = np.array([[8.2, 0.73, 0.21, 1.7, 0.074, 5, 13, 0.9968, 3.2, 0.52, 9.5, 5]])
#
# predict = model.predict(data)
# index = np.argmax(predict)
#
# wines = ['Red Wine', 'White Wine']
# print(f'\n와인의 종류는 {wines[index]} 입니다.')

