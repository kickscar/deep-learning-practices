# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: TwoLayerNet
# Estimation: Model Fitting History
import os
import pickle
from matplotlib import pyplot as plt

train_loss_file = os.path.join(os.getcwd(), 'model', 'hisotry.pkl')

train_losses = None
with open(train_loss_file, 'rb') as f:
    train_losses = pickle.load(f)

plt.plot(train_losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.show()
