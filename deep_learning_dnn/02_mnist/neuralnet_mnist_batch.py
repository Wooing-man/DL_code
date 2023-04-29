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

    batch_size = 100 # 배치 크기
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1) # 확률이 가장 높은 원소의 인덱스를 얻는다.
        accuracy_cnt += np.sum(p == y[i:i+batch_size])

    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
