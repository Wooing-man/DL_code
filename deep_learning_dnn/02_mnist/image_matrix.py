# 4강. MNIST
import sys, os
sys.path.append(os.getcwd()) # 부모 디렉터리와 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist

np.set_printoptions(linewidth=150, threshold=1000)

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

print(x_train[0])
print(y_train[0])