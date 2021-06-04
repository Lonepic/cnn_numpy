import copy
import time
import numpy as np
import scipy.io as scio


def get_data(train_pattern=""):
    f = scio.loadmat("digits.mat")
    if train_pattern == "train":
        data_x, data_y = f.get("X"), f.get("y")
    elif train_pattern == "valid":
        data_x, data_y = f.get("Xvalid"), f.get("yvalid")
    elif train_pattern == "test":
        data_x, data_y = f.get("Xvalid"), f.get("yvalid")
    else:
        print("ERROR")
        return None
    return data_x, data_y


def cross_entropy_error(y, t):
    """y:train_set_labels
        t:prediction"""
    m = y.shape[1]
    delta = 1e-7
    return -(1 / m) * np.sum(t * np.log(y + delta), keepdims=True)


def softmax(a):
    c = np.max(a)  # 防止溢出
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=0, keepdims=True)
    y = exp_a / sum_exp_a
    return y


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    n, c, h, w = input_data.shape
    out_h = (h + 2 * pad - filter_h) // stride + 1  # //表示整数除法
    out_w = (w + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((n, c, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose([0, 4, 5, 1, 2, 3]).reshape(n * out_h * out_w, -1)  # N个h*w的卷积一共要进行N*h*w此卷积运算
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def conv(conv_x, conv_y, pad, stride):
    FN, FC, FH, FW = conv_x.shape
    N, C, H, W = conv_y.shape
    out_h = 1 + int((H + 2 * pad - FH) / stride)
    out_w = 1 + int((W + 2 * pad - FW) / stride)
    col = im2col(conv_y, FH, FW, stride, pad)
    col_x = conv_x.reshape(FN, -1).T
    out = np.dot(col, col_x)
    out = out.reshape((N, out_h, out_w, -1)).transpose(0, 3, 1, 2)

    return out


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间数据（backward时使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)  # (12800, 9)
        col_W = self.W.reshape(FN, -1).T  # (9, 16)

        out = np.dot(col, col_W) + self.b  # (12800, 16)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)  # (50, 16, 16, 16)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose((1, 0)).reshape((FN, C, FH, FW))

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape((N, out_h, out_w, C)).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


class ReLU:
    def __init__(self):
        self.input = None
        self.err = None

    def forward(self, in_put):
        self.input = in_put
        return np.maximum(self.input, 0)

    def backward(self, err):
        self.err = err
        self.err[self.input < 0] = 0

        return self.err


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1 - self.out) * self.out

        return dx


class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1 - self.out ** 2)

        return dx


class MyNetwork:
    def __init__(self, input_size, conv_params, batch_size, hidden_size,
                 output_size, lr, lr_decay, weight_scale, l2, fine_tuning_epoch):
        self.conv_params = conv_params
        filter_num = conv_params['filter_num']
        filter_size = conv_params['filter_size']
        filter_stride = conv_params['stride']
        filter_pad = conv_params['pad']

        conv_output_size = int(input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size ** 0.5) / 2 * (conv_output_size ** 0.5) / 2)

        self.batch_size = batch_size

        self.pool_output_size = pool_output_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.lr = lr  # 学习率
        self.lr_decay = lr_decay
        self.weight_scale = weight_scale
        self.l2 = l2  # l2正则化参数
        self.epoch = 0
        self.fine_tuning_epoch = fine_tuning_epoch

        self.x = None

        self.w1_last = None
        self.w2_last = None
        self.conv_w_last = None

        self.h1 = None
        self.dz2 = None
        self.x_fc = None

        self.net_params = {}

        self.loss = []
        self.train_acc = []
        self.test_acc = []

        self.flag_init_weight = False
        if not self.flag_init_weight:
            self.init_weights()

        self.before_training_flag = False
        self.fine_tuning_flag = False
        self.flag = True

        # self.sigmoid = Sigmoid()
        self.tanh = Tanh()

        # 创建卷积层
        self.convolution = Convolution(
            W=self.conv_params['conv_w'],
            b=self.conv_params['conv_b'],
            stride=self.conv_params['stride'],
            pad=self.conv_params['pad']
        )

        # 创建池化层
        self.pooling = Pooling(pool_h=2, pool_w=2, stride=2)

        self.Relu = ReLU()

    def init_weights(self):
        self.conv_params['conv_w'] = np.random.randn(self.conv_params['filter_num'],
                                                     1,
                                                     self.conv_params['filter_size'],
                                                     self.conv_params['filter_size']) * self.weight_scale
        self.conv_params['conv_b'] = np.random.randn(1, self.conv_params['filter_num']) * self.weight_scale

        self.net_params['W1'] = np.random.randn(self.pool_output_size, self.hidden_size) * self.weight_scale
        self.net_params['W2'] = np.random.randn(self.hidden_size, self.output_size) * self.weight_scale
        self.net_params['b1'] = np.zeros((self.hidden_size, 1)) * self.weight_scale
        self.net_params['b2'] = np.zeros((self.output_size, 1)) * self.weight_scale

        self.net_params['W2_fine_tuning'] = np.random.randn(self.hidden_size, self.output_size) * self.weight_scale
        self.net_params['b2_fine_tuning'] = np.zeros((self.output_size, 1)) * self.weight_scale
        self.flag_init_weight = True

    def forward(self, x, y, keep_prob):
        x = x.T.reshape((x.shape[1], 16, 16))
        x = np.expand_dims(x, 1).transpose([0, 1, 3, 2])  # (50, 1, 16, 16)

        self.x = x

        x_conv = self.convolution.forward(x)  # x_conv (50, 16, 16, 16)
        x_re = self.Relu.forward(x_conv)
        x_pool = self.pooling.forward(x_re)  # (50, 16, 8, 8)

        x_fc = x_pool.reshape(x.shape[0], -1).T  # (1024, 50)
        # if self.epoch <= self.fine_tuning_epoch:
        #     drop_x_fc = np.random.randn(x_fc.shape[0], x_fc.shape[1]) < keep_prob
        #     x_fc = np.multiply(x_fc, drop_x_fc)
        #     x_fc /= keep_prob

        z1 = np.dot(self.net_params['W1'].T, x_fc)  # z1 (2048, 50)
        # h1 = self.sigmoid.forward(z1)  # h1 (320, 50)
        h1 = self.tanh.forward(z1)

        if self.epoch <= self.fine_tuning_epoch:
            drop_h1 = np.random.randn(h1.shape[0], h1.shape[1]) < keep_prob  # dropout
            h1 = np.multiply(h1, drop_h1)
            h1 /= keep_prob

        z2 = np.dot(self.net_params['W2'].T, h1)  # (10, 50)
        t = softmax(z2)

        self.h1 = h1
        self.dz2 = t - y
        self.x_fc = x_fc

        loss = cross_entropy_error(y, t)
        loss += 0.5 * self.l2 * (np.sum(np.square(self.net_params['W1'])) +
                                 np.sum(np.square(self.net_params['W2'])))  # 计算损失  L2正则化
        loss = float(loss)
        self.loss.append(loss)

        y_true = np.argmax(y, axis=0) + 1  # 计算训练精度
        y_pred = np.argmax(t, axis=0) + 1
        sum_all = 0.0
        for i in range(y_pred.shape[0]):
            if y_pred[i] == y_true[i]:
                sum_all += 1
        acc = 100 * sum_all / y_pred.shape[0]

        return acc

    def backward(self, momentum):

        if self.fine_tuning_flag:
            self.net_params['W2'] = self.net_params['W2_fine_tuning']
            self.net_params['b2'] = self.net_params['b2_fine_tuning']
            self.w2_last = self.net_params['W2']
            self.lr = 1e-1
            self.lr_decay = 8e-4
            self.fine_tuning_flag = False
            self.flag = False

        dw2 = np.dot(self.dz2, self.h1.T).T / self.batch_size  # (1400, 10)
        db2 = np.sum(self.dz2, axis=1, keepdims=True) / self.batch_size  # (10, 1)
        dh1 = np.dot(self.net_params["W2"], self.dz2) / self.batch_size  # (1400, 70)
        # dz1 = self.sigmoid.backward(dh1)
        dz1 = self.tanh.backward(dh1)
        dw1 = np.dot(self.x_fc, dz1.T) / self.batch_size  # (1024, 1400)
        db1 = np.sum(dh1, axis=1, keepdims=True) / self.batch_size  # (1400, 1)
        dx = np.dot(self.net_params["W1"], dz1) / self.batch_size  # (1024, 50)

        dx = dx.T.reshape(self.batch_size, 16, 8, 8)

        dx_pooling = self.pooling.backward(dx)  # (50, 16, 16, 16)
        dx_re = self.Relu.backward(dx_pooling)

        d_conv_w = np.zeros((16, 1, 3, 3))
        d_conv_b = np.zeros((1, 16))

        for i in range(dx_re.shape[1]):
            a1 = dx_re[:, i, :, :]
            a1 = np.expand_dims(a1, axis=1)
            a2 = conv(self.x.transpose((0, 1, 3, 2)), a1.transpose((0, 1, 3, 2)), pad=1, stride=1)
            a2 = (np.sum(a2, axis=(0, 1)) / (self.batch_size ** 2)).reshape(1, 1, 3, 3)
            d_conv_w[i, :, :, :] = a2

            a3 = np.sum(a1) / a1.size
            d_conv_b[:, i] = a3

        dw2 += self.net_params['W2'] * self.l2  # L2 normalization
        dw1 += self.net_params['W1'] * self.l2
        d_conv_w += self.conv_params['conv_w'] * self.l2

        w2_copy = copy.deepcopy(self.net_params['W2'])
        w1_copy = copy.deepcopy(self.net_params['W1'])
        conv_w_copy = copy.deepcopy(self.conv_params['conv_w'])

        # 更新权重  momentum动量梯度下降
        self.net_params['W2'] -= self.lr * dw2 + momentum * (self.net_params['W2'] - self.w2_last)
        self.net_params['b2'] -= self.lr * db2

        if self.epoch <= self.fine_tuning_epoch:
            self.net_params['W1'] -= self.lr * dw1 + momentum * (self.net_params['W1'] - self.w1_last)
            self.net_params['b1'] -= self.lr * db1
            self.conv_params['conv_w'] -= self.lr * d_conv_w + momentum * \
                                          (self.conv_params['conv_w'] - self.conv_w_last)
            self.conv_params['conv_b'] -= self.lr * d_conv_b

        if self.epoch <= 100:
            self.lr = self.lr  # renew the learning rate
        elif 100 < self.epoch:
            self.lr = self.lr * (1 - self.lr_decay)

        self.w2_last = w2_copy
        if self.epoch <= self.fine_tuning_epoch:
            self.w1_last = w1_copy
            self.conv_w_last = conv_w_copy

    def train(self, train_x, train_y, momentum, target_acc):
        while True:
            self.w1_last = self.net_params['W1']
            self.w2_last = self.net_params['W2']
            self.conv_w_last = self.conv_params['conv_w']

            # for n in range(0, train_x.shape[1], self.batch_size):  # 用batch_size划分数据集
            #   x = train_x[:, n: n + self.batch_size]  # 数据集按列向量进行切片
            #   y = train_y[:, n: n + self.batch_size]
            for n in range(10):
                num_train = 5000
                indices = list(range(num_train))
                np.random.seed()
                idx = np.random.choice(indices, size=self.batch_size, replace=False)
                x = train_x[:, idx]  # 随机选择训练数据
                y = train_y[:, idx]
                self.forward(x, y, keep_prob=0.5)
                if self.epoch >= self.fine_tuning_epoch and self.flag:
                    self.fine_tuning_flag = True
                self.backward(momentum=momentum)

            train_acc = self.forward(train_x, train_y, keep_prob=1)
            if train_acc >= target_acc or self.epoch >= 250:
                print('The epoch of the phase of training is ' + str(self.epoch))
                print('The accuracy of the phase of training is ' + str(train_acc) + '%')

                return None
            if self.epoch % 5 == 0:
                print('epoch is ' + str(self.epoch) + ' accuracy is ' + str(train_acc) + '% learning rate is '
                      + str(self.lr))
            self.train_acc.append(train_acc)

            self.epoch += 1

    def test(self, test_x, test_y):
        test_acc = self.forward(test_x, test_y, keep_prob=1)
        self.test_acc.append(test_acc)
        print(('The accuracy of the phase of testing is ' + str(test_acc) + '%'))

    def valid(self, test_x, test_y):
        valid_acc = self.forward(test_x, test_y, keep_prob=1)
        print(('The accuracy of the phase of validing is ' + str(valid_acc) + '%'))

    def get_loss_history(self):
        return self.loss

    def get_acc_history(self):
        return self.train_acc, self.test_acc


if __name__ == '__main__':
    train_set_x_raw, train_set_y_raw = get_data(train_pattern="train")
    test_set_x_raw, test_set_y_raw = get_data(train_pattern="test")
    valid_set_x_raw, valid_set_y_raw = get_data(train_pattern="valid")

    train_set_x = (train_set_x_raw / 255).T
    test_set_x = (test_set_x_raw / 255).T
    valid_set_x = (valid_set_x_raw / 255).T

    train_set_y = np.zeros((10, 5000))
    for C in range(train_set_y_raw.shape[0]):
        train_set_y[train_set_y_raw[C] - 1, C] = 1

    test_set_y = np.zeros((10, 5000))
    for Z in range(test_set_y_raw.shape[0]):
        test_set_y[test_set_y_raw[Z] - 1, Z] = 1

    valid_set_y = np.zeros((10, 5000))
    for Z in range(valid_set_y_raw.shape[0]):
        valid_set_y[valid_set_y_raw[Z] - 1, Z] = 1

    conv_params = {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1}

    My_NetWork = MyNetwork(input_size=256, conv_params=conv_params, batch_size=80, hidden_size=1600, output_size=10,
                           lr=1e-1, lr_decay=8e-4, weight_scale=1, l2=1e-4, fine_tuning_epoch=251)
    # 5e-4  # hidden_size=1200 acc=87.36    1500 88
    start = time.time()

    My_NetWork.train(train_set_x, train_set_y, momentum=0.9, target_acc=96)

    My_NetWork.test(test_set_x, test_set_y)

    My_NetWork.valid(valid_set_x, valid_set_y)

    print(str(time.time() - start) + 's')
