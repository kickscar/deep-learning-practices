# coding: utf-8
# 05.mnist-neural-network(Modified National Institute of Standards and Technology) Database 손글씨
# Data Testing
# 부모 디렉터리의 파일을 가져올 수 있도록 설정
import sys
import os
import numpy as np
from PIL import Image

try:
    sys.path.append(os.path.join(os.getcwd(), 'lib'))
    from lib.mnist import load_mnist
except ImportError:
    raise ImportError("lib.mnist Module Can Not Found")


(x_train, l_train), (x_test, l_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = l_train[0]
print(label)                # 5

print(img.shape)            # (784,)
img = img.reshape(28, 28)   # 형상을 원래 이미지의 크기로 변형
print(img.shape)            # (28, 28)

# 이미지 보기 (P)ython (I)mage (L)ibrary 사용
pil_img = Image.fromarray(np.uint8(img))
pil_img.show()
