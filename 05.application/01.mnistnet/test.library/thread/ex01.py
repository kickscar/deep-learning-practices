# a slight modification of your code using multiprocessing
import multiprocessing
import time
from matplotlib import pyplot as plt, animation
import numpy as np

# Graph Process
fig, ax = plt.subplots()
line, = ax.plot([], [])


def flow(i):
    x = np.arange(0, 2 * np.pi, 0.01)
    line.set_data(x, np.sin(x + i / 50))
    return line,


def graph_process():
    ax.grid()
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1, 1)

    ani = animation.FuncAnimation(fig, flow, interval=20, blit=True, save_count=50)
    plt.show()
    time.sleep(4)


if __name__ == '__main__':
    multiprocessing.Process(target=graph_process, args=()).start()
    print(multiprocessing.current_process().name, "- The main process is continuing...")

