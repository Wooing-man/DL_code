# 5강. 신경망 구현
import sys, os
sys.path.append(os.getcwd())
import numpy as np
import pickle
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, flatten=True,
    one_hot_label=False)
    return x_test, y_test


def init_network():
    with open("./ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


if __name__ == '__main__':
    x, y = get_data()
    network = init_network()
    error = []
    for i in range(len(x)):
        y_hat = predict(network, x[i])
        p = np.argmax(y_hat) # 확률이 가장 높은 원소의 인덱스를 얻는다.
        if p != y[i]:
            error.append(i)

    print(error)

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x[error[i]].reshape(28, 28))
        plt.xlabel(y[error[i]])

    plt.show()