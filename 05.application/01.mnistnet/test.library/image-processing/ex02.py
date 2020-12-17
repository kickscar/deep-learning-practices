# OpenCV Grayscale Image Handling
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_origin = cv2.imread('../../images/number.png', cv2.IMREAD_GRAYSCALE)
print(img_origin)

# resize(28 x28)
img_resize = cv2.resize(img_origin, (28, 28))
print(img_resize.shape)

# invert
img_invert = cv2.bitwise_not(img_resize)
print(img_invert)

# flatten(matrix to vector)
img_flatten = img_invert.reshape(-1)
print(img_flatten.shape)

# nomalize
img_mormalize = img_flatten.astype(np.float32)
img_mormalize /= 255.0
print(img_mormalize)


# view
ax0 = plt.subplot(1, 2, 1)
ax0.imshow(img_origin, cmap='gray')

ax0 = plt.subplot(1, 2, 2)
ax0.imshow(img_invert, cmap='gray')

plt.show()



