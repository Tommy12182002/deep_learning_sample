# coding: utf-8
# neural_network_mnist.pyのバッチver(高速)
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.functions import softmax, sigmoid
import numpy as np
import pickle

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=False, flatten=True)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        new_network = pickle.load(f)
    return new_network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y  = softmax(a3)
    return y

# 入力データと答えのラベル（0~10）
x, t = get_data()
# 重みとバイアス
network    = init_network()
batch_size = 100
# 予測の精度
accuracy_cnt = 0

n = len(x)

for i in range(0, n, batch_size):
    x_batch = x[i: i + batch_size]
    y_batch = predict(network, x_batch)

    # 確率で一番高いものが予測で導き出された答え
    p      = np.argmax(y_batch, axis=1)
    temp_t = t[i:i+batch_size]
    accuracy_cnt += np.sum(p == temp_t)

print("予測の精度は" + str(float(accuracy_cnt / n) * 100) + "%")
