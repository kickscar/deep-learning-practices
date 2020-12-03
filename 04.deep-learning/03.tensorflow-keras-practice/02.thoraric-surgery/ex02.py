# coding: utf-8
# Thoraric Surgery Prediction Model
# Model Fitting
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np

# 1. load training/test data
Data_set = np.loadtxt("./dataset/thoraric-surgery.csv", delimiter=",")
X = Data_set[:, 0:17]
Y = Data_set[:, 17]

# 2. model frame config
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. model fitting environment
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. model fitting
history = model.fit(X, Y, epochs=100, batch_size=10)

# 5. training loss
train_loss = history.history['loss']

# 6. 그래프로 표현
xlen = np.arange(len(train_loss))
plt.plot(xlen, train_loss, marker='.', c="blue", label='train loss')
plt.show()
