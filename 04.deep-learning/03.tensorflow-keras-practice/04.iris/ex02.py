# coding: utf-8
# Iris Species Multi-Class Classification Model
# Model Fitting
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# 1. load training/test data
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

df = pd.read_csv('./dataset/iris.csv', names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
dataset = df.values

train_x = dataset[:, 0:4].astype(float)

train_t = dataset[:, 4]
e = LabelEncoder()
e.fit(train_t)
train_t = e.transform(train_t)
train_t = tf.keras.utils.to_categorical(train_t)

# 2. model frame config
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. model fitting environment
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. model fitting
history = model.fit(train_x, train_t, epochs=50, batch_size=1, verbose=1)

# 5. result
result = model.evaluate(train_x, train_t, verbose=0)
print(f'\n (Loss, Accuracy) = ({result[0]}, {result[1]})')
