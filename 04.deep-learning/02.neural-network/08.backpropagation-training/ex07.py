# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: TwoLayerNet
# Estimation: Training Accuracy(Overfitting)
import os
import pickle
from matplotlib import pyplot as plt

train_accuracy_file = os.path.join(os.getcwd(), 'dataset', 'twolayer_train_accuracy.pkl')
test_accuracy_file = os.path.join(os.getcwd(), 'dataset', 'twolayer_test_accuracy.pkl')

train_accuracies = None
test_accuracies = None
with open(train_accuracy_file, 'rb') as f_train_accuracy, open(test_accuracy_file, 'rb') as f_test_accuracy:
    train_accuracies = pickle.load(f_train_accuracy)
    test_accuracies = pickle.load(f_test_accuracy)

plt.plot(train_accuracies, label='train accuracy')
plt.plot(test_accuracies, label='test accuracy')

plt.xlim(0, 20, 1)
plt.ylim(0., 1., 0.5)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')

plt.show()
