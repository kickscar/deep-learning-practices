# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: MNISTNet
# Prediction from Model Saved
import pickle
import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    import multilayernet as network
except ImportError:
    raise ImportError("module can not be found")


def main():
    axes = graph_setup()

    params = model_load()
    model_frame(784, 10, (50, 100), params)

    image_file = os.path.join(Path(os.getcwd()).parent, 'images', 'number.png')
    watch_file_modified(image_file, *axes)


def graph_setup():
    fig = plt.figure(figsize=(5, 3))
    spec = plt.GridSpec(20, 20, bottom=0.22, top=0.85, left=0.09, right=0.95, hspace=0, wspace=0)

    ax0 = fig.add_subplot(spec[:1, :-1])
    ax1 = fig.add_subplot(spec[2:, :9], xlabel='Original')
    ax2 = fig.add_subplot(spec[2:, 11:], xlabel='Inverted')

    ax0.set_title(label="Handwritten Digit Prediction", )
    ax0.axis("off")

    plt.draw()
    plt.pause(0.01)

    return ax1, ax2


def model_load():
    model_file = os.path.join(Path(os.getcwd()).parent, 'model', 'model.pkl')

    with open(model_file, 'rb') as f:
        params = pickle.load(f)

    return params


def model_frame(input_size, output_size, hidden_size=(10, ), params=None):
    network.initialize(input_size=input_size, hidden_sizes=hidden_size, output_size=output_size, init_params=params)


def image_load(image_file):
    img_origin = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img_origin, (28, 28))          # resize(28x28)
    img_inverted = cv2.bitwise_not(img_resized)             # inverted
    img_flatten = img_inverted.reshape(-1)                  # flatten(matrix to vector)
    img_normalized = img_flatten.astype(np.float32) / 255.  # normalized

    return img_origin, img_inverted, img_normalized


def watch_file_modified(image_file, ax1, ax2):
    timestamp_file_modified = os.stat(image_file).st_mtime

    try:
        while True:
            time.sleep(1)

            timestamp = os.stat(image_file).st_mtime

            if timestamp_file_modified != timestamp:
                # load image
                origin, inverted, normalized = image_load(image_file)

                # draw image
                ax1.imshow(origin, cmap='gray')
                ax2.imshow(inverted, cmap='gray')

                plt.draw()
                plt.pause(0.01)

                # prediction
                val, y = network.predict(normalized)

                print(f'Model Prediction: {val}\n')
                print('Class Probabilities')
                for idx, probablity in enumerate(np.round(y*100., 2)):
                    print(f'{idx}: {probablity:5.2f}%')

                # timestamp cached
                timestamp_file_modified = timestamp

    except KeyboardInterrupt:
        print('\nDone')


__name__ == '__main__' and main()
