# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error, sigmoid
from common.gradient  import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        # 1層目
        W1 = self.params['W1']
        b1 = self.params['b1']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        # 2層目
        W2 = self.params['W2']
        b2 = self.params['b2']
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)

        # 出力層
        return softmax(z2)

    # 損失を取得
    def loss(self, x, t):
        return cross_entropy_error(self.predict(x), t)

    # 予測精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

# 2層のニューラルネットワークを対象に、勾配を求める
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
x   = np.random.rand(100, 784)
t   = np.random.rand(100, 10)
grads = net.numerical_gradient(x, t)
print(grads)
