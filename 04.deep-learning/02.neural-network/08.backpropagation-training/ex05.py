# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: MultiLayerNet
# Estimation: Model Fitting History
import os
import pickle
from matplotlib import pyplot as plt

history = None

history_file = os.path.join(os.getcwd(), 'model', 'history.pkl')
with open(history_file, 'rb') as f:
    history = pickle.load(f)

plt.plot(history['accuracy'], marker='.', c='blue', label='Train Accuracy')
plt.plot(history['val_accuracy'], marker='.', c='red', label='Test Accuracy')
plt.plot(history['loss'], marker='.', c='blue', label='Train Loss')
plt.plot(history['val_loss'], marker='.', c='red', label='Test Loss')

plt.xlabel('Iteration')
plt.ylabel('Loss - Accuracy')
plt.legend(loc='best')

plt.grid()

plt.show()
