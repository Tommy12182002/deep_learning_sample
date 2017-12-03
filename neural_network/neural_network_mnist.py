# coding: utf-8
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
network = init_network()
# 予測の精度
accuracy_cnt = 0

n = len(x)

for i in range(n):
    # 予測後のデータは添字0~9の配列で返却され、それぞれ確率が含まれている
    y = predict(network, x[i])
    # 確率で一番高いものが予測で導き出された答え
    p = np.argmax(y)

    if p == t[i]:
        accuracy_cnt += 1

print(accuracy_cnt)
print(n)
print("予測の精度は" + str(float(accuracy_cnt / n)))
