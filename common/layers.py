# coding: utf-8

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

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x * 0.5) * 0.5 + 0.5
        return self.out

    def backword(self, dout):
        return dout * (1.0 - self.out) * self.out

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b

        self.x = None
        # この変数はテンソル対応
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        # 行列に変換する
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx
