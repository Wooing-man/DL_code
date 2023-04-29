# 4강. MNIST
import sys, os
sys.path.append(os.getcwd()) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = y_train[0]
print(label)

print(img.shape) # (784,)
img = img.reshape(28, 28) # 형상을 원래 이미지의 크기로 변형
print(img.shape) # (28, 28)

img_show(img)