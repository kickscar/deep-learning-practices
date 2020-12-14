# coding: utf-8
# Training Neural Network
# Data Set: MNIST Handwritten Digit Data Set
# Network: MultiLayerNet
# Prediction from Model Saved
import pickle
import sys
import os
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import multilayernet as network
except ImportError:
    raise ImportError("Library Module Can Not Found")

# 1. load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. load model
model_file = os.path.join(os.getcwd(), 'model', 'model.pkl')
params = None
with open(model_file, 'rb') as f:
    params = pickle.load(f)

# 3. model frame
input_size, output_size = train_x.shape[1], train_t.shape[1]
network.initialize(input_size=input_size, hidden_sizes=[50, 100], output_size=output_size, init_params=params)


# 4. load image
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
