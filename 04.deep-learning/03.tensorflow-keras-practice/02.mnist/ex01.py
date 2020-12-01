# coding: utf-8
# MNIST handwritten digit classification model
import sys
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Dense
try:
    sys.path.append(os.path.join(Path(os.getcwd()), 'lib'))
    from mnist import load_mnist
except ImportError:
    raise ImportError("Library Module Can Not Found")

# 1. load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. model frame config
model = Sequential()
model.add(Dense(50, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. model fitting environment
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. model fitting
history = model.fit(
    train_x,
    train_t,
    validation_data=(test_x, test_t),
    epochs=30,
    batch_size=100,
    verbose=1)

# checkpointer = ModelCheckpoint(filepath='./dataset/{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
# early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
# history = model.fit(
#     train_x,
#     train_t,
#     validation_data=(test_x, test_t),
#     epochs=30,
#     batch_size=100,
#     verbose=1,
#     callbacks=[early_stopping_callback, checkpointer])


# 5. training loss
train_loss = history.history['loss']
test_loss = history.history['val_loss']

# 6. 그래프로 표현
x_len = np.arange(len(train_loss))
plt.plot(x_len, train_loss, marker='.', c="blue", label='train loss')
plt.plot(x_len, test_loss, marker='.', c="red", label='test loss')
plt.show()
