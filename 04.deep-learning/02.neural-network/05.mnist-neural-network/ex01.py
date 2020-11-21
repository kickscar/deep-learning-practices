# coding: utf-8
# MNIST(Modified National Institute of Standards and Technology)
# MNIST 손글씨 숫자 분류 신경망 (Neural Network for MNIST Handwritten Digit Classification): 데이터 살펴보기
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
except ImportError:
    raise ImportError("Library Module Can Not Found")


(train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=False, flatten=True, one_hot_label=False)

image = train_images[0]
label = train_labels[0]
print(label)                    # 5

print(image.shape)              # (784,)
image = image.reshape(28, 28)   # 형상을 원래 이미지의 크기로 변형
print(image.shape)              # (28, 28)

# 이미지 보기: PIL(Python Image Library) 사용
pil_img = Image.fromarray(np.uint8(image))
pil_img.show()
