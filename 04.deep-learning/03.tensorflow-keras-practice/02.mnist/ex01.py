# coding: utf-8
# MNIST handwritten digit classification model
import sys
import os
from pathlib import Path
from tensorflow.python.keras import Sequential
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
model.fit(train_x, train_t,  epochs=30, batch_size=100, verbose=1)
