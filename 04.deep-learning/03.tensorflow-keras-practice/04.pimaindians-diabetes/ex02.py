import numpy as np
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
model.fit(x, t, epochs=200, batch_size=10)

# 5. result
# print("\n Accuracy: %.4f" % (model.evaluate(x, t)[1]))
