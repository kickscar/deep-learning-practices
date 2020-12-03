# coding: utf-8
# Sonar Mineral Binary Classification Model
# Model Fitting #2 - Overfitting
import os
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.python.keras.utils.np_utils import to_categorical

# 1. load training/test data
df = pd.read_csv('./dataset/sonar.csv', header=None)

dataset = df.values
x = dataset[:, 0:60].astype(float)

t = dataset[:, 60]
e = LabelEncoder()
e.fit(t)
t = e.transform(t)
t = to_categorical(t)

# 2. split train & test
# print(x.shape, t.shape)
train_x, test_x, train_t, test_t = train_test_split(x, t, test_size=0.3, random_state=0)
# print(train_x.shape, train_t.shape)
# print(test_x.shape, test_t.shape)

# 2. model frame config
model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

# 3. model fitting environment
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. model fitting
model.fit(train_x, train_t, epochs=200, batch_size=5, verbose=1)

# 5. result
result = model.evaluate(train_x, train_t, verbose=0)
print(f'\n (Train Loss, Train Accuracy) = ({result[0]}, {result[1]})')

# 6. save model
model_directory = os.path.join(os.getcwd(), 'model')
if not os.path.exists(model_directory):
    os.mkdir(model_directory)

model.save(os.path.join(model_directory, 'model.h5'))

# 7. test
del model
model = load_model(os.path.join(model_directory, 'model.h5'))

result = model.evaluate(test_x, test_t, verbose=0)
print(f'\n (Test Loss, Test Accuracy) = ({result[0]}, {result[1]})')
