# coding: utf-8
import numpy as np

# 数値微分を求める
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return x[0]**2 + x[1]**2
    # または return np.sum(x**2)

x = np.arange(0.0, 20.0, 0.1)
y = numerical_diff(function_1, x)
print(y)
