# coding: utf-8
# Pima Indians Diabets Prediction Model
# Model Fitting
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

# 1. load training/test data
dataset = np.loadtxt("./dataset/pimaindians-diabetes.csv", delimiter=",")
x = dataset[:, 0:8]
t = dataset[:, 8]

# 2. model frame config
model = Sequential()
model.add(Dense(50, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. model fitting environment
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. model fitting
history = model.fit(x, t, epochs=200, batch_size=10)

# 5. training loss
train_loss = history.history['loss']

# 6. 그래프로 표현
x_len = np.arange(len(train_loss))
plt.plot(x_len, train_loss, marker='.', c="blue", label='train loss')
plt.show()

# 5. result
# 5. result
result = model.evaluate(x, t, verbose=0)
print(f'\n (Loss, Accuracy) = ({result[0]}, {result[1]})')