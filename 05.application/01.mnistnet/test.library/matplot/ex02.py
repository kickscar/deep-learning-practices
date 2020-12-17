# OpenCV Grayscale Image Handling
import time
import cv2
import matplotlib.pyplot as plt

img_origin = cv2.imread('../../images/number.png', cv2.IMREAD_GRAYSCALE)

fig, ax = plt.subplots()
ax.axis("off")
plt.draw()
plt.pause(0.01)

img_resize = cv2.resize(img_origin, (28, 28))
img_invert = cv2.bitwise_not(img_origin)

print('ready to start inverting show...')
time.sleep(3)

for idx in range(0, 10):
    ax.imshow(img_resize if idx % 2 == 0 else img_invert, cmap='gray')
    plt.draw()
    plt.pause(0.01)

    print(f'{idx}.......')
    time.sleep(1)

plt.show()


