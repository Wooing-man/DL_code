# 4강. MNIST
import sys, os
sys.path.append(os.getcwd()) # 부모 디렉터리와 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=150, threshold=1000)

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

plt.figure()
plt.imshow(x_train[0][0])
plt.colorbar()
plt.show()

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    # plt.imshow(x_train[i][0], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])

plt.show()