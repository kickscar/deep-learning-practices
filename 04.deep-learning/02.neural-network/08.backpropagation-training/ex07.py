# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: MultiLayerNet
# Estimation: Training Accuracy(Overfitting)
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np

train_accuracy_file = os.path.join(os.getcwd(), 'model', 'train_accuracy.pkl')
test_accuracy_file = os.path.join(os.getcwd(), 'model', 'test_accuracy.pkl')

train_accuracies = None
test_accuracies = None
with open(train_accuracy_file, 'rb') as f_train_accuracy, open(test_accuracy_file, 'rb') as f_test_accuracy:
    train_accuracies = pickle.load(f_train_accuracy)
    test_accuracies = pickle.load(f_test_accuracy)

xlen = np.arange(len(train_accuracies))
plt.plot(xlen, train_accuracies, marker='.', c="blue", label='train accuracy')
plt.plot(xlen, test_accuracies, marker='.', c="red", label='test accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')

plt.ylim(0., 1., 0.5)

plt.show()
