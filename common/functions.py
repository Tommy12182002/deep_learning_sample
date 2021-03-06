# coding: utf-8
import numpy as np

# --------------------------------------
# 活性化関数
# --------------------------------------
def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    # return  1 / (1 + np.exp(-x))
    # overflowするのでexpの代わりにtanhを使う
    return np.tanh(x * 0.5) * 0.5 + 0.5

def relu(x):
    return np.maximum(0, x)

# --------------------------------------
# 出力層の関数
# --------------------------------------
def softmax(x):
    # 2次元配列の場合は
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

def identity_function(val):
    return val

# --------------------------------------
# 損失関数
# --------------------------------------
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    # 例) [60000,10]の配列(10は0,0,0,0,0,1,0,0,0,0)なら、[60000]に変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
