import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


f = plt.figure(figsize=(15, 5))
g = plt.GridSpec(1, 2, hspace=0, wspace=0)

ax1 = f.add_subplot(g[0, 0])

x = np.arange(0, 2 * np.pi, 0.01)
line1, = ax1.plot(x, np.sin(x))

ax2 = plt.subplot(1, 2, 2)


def animate(i):
    line1.set_ydata(np.sin(x + i / 50))
    return line1,


ani = animation.FuncAnimation(f, animate, interval=20, blit=True, save_count=50)


plt.show()

