# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from common.functions import softmax, cross_entropy_error, sigmoid
from common.gradient  import numerical_gradient

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=False, one_hot_label=True)

train_loss_list = []

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0] # 60000とか
batch_size = 100
learning_rate = 0.1

train_acc_list = []
test_acc_list = []
# 1 エポックあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 0~60000のうちのランダムな数値を100こ抽出
    batch_mask = np.random.choice(train_size, batch_size)
    # x_batch, t_batchは(batch_mask, 784)の配列
    x_batch    = x_train[batch_mask]
    t_batch    = t_train[batch_mask]

    grads = network.numerical_gradient(x_batch, t_batch)

    # パラメータ更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grads[key]

    # 学習通過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1 エポックごとに認識精度を計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc  = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

print('done')
