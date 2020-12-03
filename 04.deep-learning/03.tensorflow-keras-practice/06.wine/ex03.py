# coding: utf-8
# Wine Binary Classification Model(와인 종류 분류 모델)
# Model Fitting #2 - model update
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint
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

# 4. model check point config
model_directory = os.path.join(os.getcwd(), 'model')
if not os.path.exists(model_directory):
    os.mkdir(model_directory)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(model_directory, '{epoch:02d}-{val_loss:.4f}.hdf5'),
    monitor='val_loss',     # val_loss(테스트셋 오차), loss(학습셋 오차), val_acc(테스트셋 전확도), acc(학습셋 정확도)
    verbose=1,
    save_best_only=True)

# 5. model fitting
model.fit(x, t, validation_split=0.2, epochs=200, batch_size=200, verbose=0, callbacks=[checkpoint])

# 6. result
result = model.evaluate(x, t, verbose=0)
print(f'\n(Loss, Accuracy) = ({result[0]}, {result[1]})')


