# 5강. 신경망 구현
import sys, os
sys.path.append(os.getcwd())
import numpy as np
import pickle
from dataset.mnist import load_mnist


with open("./ch03/sample_weight.pkl", "rb") as f:
    network = pickle.load(f)

w1, w2, w3 = network['W1'], network['W2'], network['W3']
b1, b2, b3 = network['b1'], network['b2'], network['b3']

print(w1)

print(type(network))
print(network.keys())
print('w1 shape:' + str(w1.shape))
print('w2 shape:' + str(w2.shape))
print('w3 shape:' + str(w3.shape))
print('b1 shape:' + str(b1.shape))
print('b2 shape:' + str(b2.shape))
print('b3 shape:' + str(b3.shape))