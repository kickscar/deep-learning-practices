# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: MNISTNet
# Prediction from Model Saved
import pickle
import sys
import os
import numpy as np
from PIL import Image, ImageOps
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    import multilayernet as network
except ImportError:
    raise ImportError("Library Module Can Not Found")

# 1. load model
model_file = os.path.join(os.getcwd(), 'model', 'model.pkl')
params = None
with open(model_file, 'rb') as f:
    params = pickle.load(f)

# 2. model frame
input_size, output_size = 784, 10
network.initialize(input_size=input_size, hidden_sizes=[50, 100], output_size=output_size, init_params=params)

# 3. load image
im = ImageOps.invert(Image.open('images/test.bmp'))
w, h = im.size

pixels = np.zeros((h, w), dtype=np.uint8)

for r in range(0, h):
    for c in range(0, w):
        pixels[r, c] = np.average(im.getpixel((c, r)))

# flatten(matrix to vector)
pixels = pixels.reshape(-1)

# nomalize
pixels = pixels.astype(np.float32)
pixels /= 255.0

# 5. predict
y = network.predict(pixels)
print(f'predict: {y}')
