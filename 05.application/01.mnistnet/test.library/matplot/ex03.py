# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: MultiLayerNet
# Estimation: Model Fitting History
import os
import pickle
import time
from pathlib import Path

from matplotlib import pyplot as plt

history = {
    'accuracy': [],
    'loss': []
}

# history_file = os.path.join(Path(os.getcwd()).parent.parent, 'model', 'history.pkl')
# with open(history_file, 'rb') as f:
#     history = pickle.load(f)

f = plt.figure(figsize=(15, 5))
g = plt.GridSpec(1, 2, hspace=0, wspace=0)

ax1 = f.add_subplot(g[0, 0], xlabel='Iteration', ylabel='Loss - Accuracy')

line1 = ax1.plot([])[0]
line1.set_marker('.')
line1.set_color('red')
line1.set_label('Train Accuracy')

line2 = ax1.plot([])[0]
line2.set_marker('.')
line2.set_color('blue')
line2.set_label('Train Loss')


ax1.legend(handles=[line1, line2], loc='best')
ax1.grid()
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 2)

plt.draw()
plt.pause(0.01)

for idx in range(0, 10):
    history['accuracy'].append((idx, idx*0.2))
    history['loss'].append((idx, idx*0.1))

    line1.set_data(zip(*history['accuracy']))
    line2.set_data(zip(*history['loss']))

    plt.draw()
    plt.pause(0.01)

    time.sleep(1)

plt.show()



