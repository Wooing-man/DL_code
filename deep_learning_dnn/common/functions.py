import numpy as np


def identify_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def softmax(x):
    if x.ndim == 2: # 배치일 때
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, y_hat):
    return 0.5 * np.sum((y-y_hat)**2)


def cross_entropy_error(y, y_hat):
    if y.ndim == 1: # 벡터형태 일 때,
        y = y.reshape(1, y.size) # 원-핫 인코딩으로 변환
        y_hat = y_hat.reshape(1, y_hat.size) # 원-핫 인코딩으로 변환

    # 훈련데이터가 원-핫 인코딩 형태의 벡터라면, 정답 레이블의 인덱스로 변환
    if y.size == y_hat.size:
        y_hat = y_hat.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), y_hat] + 1e-7)) / batch_size

