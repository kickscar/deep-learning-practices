# coding: utf-8
# Sonar Mineral Binary Classification Model
# Model Fitting #1
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# 1. load training/test data
df = pd.read_csv('./dataset/sonar.csv', header=None)

dataset = df.values
x = dataset[:, 0:60].astype(float)

t = dataset[:, 60]
e = LabelEncoder()
e.fit(t)
t = e.transform(t)

# 2. model frame config
model = Sequential()
model.add(Dense(24,  input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. model fitting environment
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# 4. model fitting
model.fit(x, t, epochs=200, batch_size=5, verbose=1)

# 5. result
result = model.evaluate(x, t, verbose=0)
print(f'\n (Loss, Accuracy) = ({result[0]}, {result[1]})')


