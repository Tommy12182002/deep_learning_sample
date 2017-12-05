# coding: utf-8
import numpy as np
from numerical_gradient import numerical_gradient

# 分配降下法
# @lr: 学習率
# @step_num: 繰り返し数
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

def function_2(x):
    return x[0]**2 + x[1]**2


init_x = np.array([-3.0, 4.0])
y      = gradient_descent(function_2, init_x=init_x, lr=10, step_num=100)

print(y)
