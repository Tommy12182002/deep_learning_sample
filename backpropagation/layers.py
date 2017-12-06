# coding: utf-8
class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forword(self, x, y):
        self.x = x
        self.y = y
        return x + y

    def backword(self, dout):
        return dout * 1, dout * 1

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forword(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backword(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

class Relu:
    def __init__(self):
        self.mask = None

    def forword(self, x):
        self.mask = (x <= 0)
        out       = x.copy()
        # trueになる要素にのみ0を代入する
        out[self.mask] = 0
        return out

    def backword(self, dout):
        dout[self.mask] = 0
        return dout
