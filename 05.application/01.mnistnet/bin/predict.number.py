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
import curses
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
    stdscr, = curses_init()

    params = model_load()
    model_frame(784, 10, (50, 100), params)

    image_file = os.path.join(Path(os.getcwd()).parent, 'images', 'number.png')
    watch_file_modified(image_file, *axes, stdscr)


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


def curses_init():
    stdscr = curses.initscr()

    stdscr.keypad(True)
    curses.noecho()
    curses.cbreak()
    curses.curs_set(False)

    curses.start_color()
    curses.use_default_colors()

    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)

    return stdscr,


def curses_cleanup(stdscr):
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()


def curses_prediction_result(stdscr, val, y):
    stdscr.addstr(0, 0, f'Model Prediction:', curses.color_pair(1))
    stdscr.addstr(0, 18, f'{val}', curses.color_pair(3) | curses.A_BOLD | curses.A_UNDERLINE)
    stdscr.addstr(0, 25, 'Class Probabilities', curses.color_pair(1))

    for idx, probablity in enumerate(np.round(y * 100., 2)):
        color_pair = curses.color_pair(3) if idx == val else curses.color_pair(2)
        filled_length = int(30 * probablity // 100)
        bar = '█' * filled_length + '-' * (30 - filled_length)

        stdscr.addstr(1+idx, 25, f'{idx}', color_pair | curses.A_BOLD | curses.A_UNDERLINE)
        stdscr.addstr(1+idx, 26, f': {probablity:5.2f}%', color_pair)
        stdscr.addstr(1+idx, 35, f'|{bar}|', color_pair)

    stdscr.refresh()


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


def watch_file_modified(image_file, ax1, ax2, stdscr):
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
                prediction_result = network.predict(normalized)
                curses_prediction_result(stdscr, *prediction_result)

                # timestamp cached
                timestamp_file_modified = timestamp

    except KeyboardInterrupt:
        pass
    finally:
        curses_cleanup(stdscr)


__name__ == '__main__' and main()
