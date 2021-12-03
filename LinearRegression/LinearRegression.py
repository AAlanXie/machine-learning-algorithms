"""
coding: utf-8
author: tianqi
Email: tianqixie98@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 自定义线性回归的类
class LinearRegression:
    # 构造函数
    def __init__(self):
        self.w = None
        self.loss_history = []

    # 训练函数
    def fit(self, x, y, iteration=1000, learning_rate=0.001, method='sgd'):
        """
        :param x: 输入特征向量
        :param y: 输出特征向量
        :param iteration: 迭代次数
        :param learning_rate: 学习速率
        :param method: 训练方法
        :return: 没有返回
        """
        # 定义样本数和特征数
        num_of_samples, num_of_features = x.shape
        # 初始化权重
        self.w = np.random.rand(num_of_features, 1)
        # 初始化损失值
        self.loss_history = []
        # 开始迭代
        for i in range(iteration):
            if method == 'sgd':
                # 随机选择一个样本进行计算梯度和损失值
                idx = np.random.choice(num_of_samples)
                # 注意这里必须加上np.newaxis，否则会报维度不同的错误
                loss, grad = loss_and_grad(x[idx, np.newaxis], y[idx, np.newaxis], self.w)
                self.w = self.w - learning_rate * grad
                self.loss_history.append(loss)
                if i % 100 == 0:
                    per = (i + 1) / iteration * 1.0
                    print(f'{per}% has finished!!')
            else:
                # 正常梯度下降
                loss, grad = loss_and_grad(x, y, self.w)
                self.w = self.w - learning_rate * grad
                self.loss_history.append(loss)
                if i % 100 == 0:
                    per = (i + 1) / iteration * 1.0
                    print(f'{per}% has finished!!')

    # 预测
    def predict(self, x_pred):
        """
        :param x_pred: 需要预测的x
        :return: 预测后的y值
        """
        y_pred = x_pred.dot(self.w)
        return y_pred

    # 显示损失曲线
    def show_loss(self):
        plt.plot([i for i in range(len(self.loss_history))], self.loss_history)
        plt.show()

    def show_param(self):
        return self.w

def loss_and_grad(x, y, w):
    """
    :param x: 对应的特征输入空间  形式为[1, x]
    :param y: 对应的特征输出空间  形式为[y]
    :param w: 对应的权重(将b纳入，设置成为w0)
    :return:
    """
    f = x.dot(w)
    diff = f - y
    loss = sum(diff * diff) / 2
    grad = 2 * (x.T).dot(diff)
    return loss, grad

if __name__ == '__main__':
    # 初始化数据，自定义一些多元的线性回归数据
    x = np.random.randint(0, 10, size=[50, 4])
    w_init = np.array([0.3, 0.4, 0.6, 0.5]).reshape(-1, 1)
    y = x.dot(w_init) + np.random.rand(50, 1)
    # 自定义类，实现多元函数后输出损失函数，同时也可以查看对应的w
    model = LinearRegression()
    model.fit(x, y)
    # model.show_loss()
    print(model.show_param())


