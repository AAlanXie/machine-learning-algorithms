"""
coding: utf-8
author: tianqi
email: tianqixie98@gmail.com
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import time
import math

"""
模型比sklearn中的模型运算效率低非常多，sklearn中的模型优化在哪里了呢？
不用对于每个预测数据都进行遍历，直接向量化？还是使用kd_tree等算法优化了呢？
"""


class MyKNN:
    def __init__(self):
        self.x = None
        self.y = None

    def fit(self, x, y):
        """
        :desc 对于KNN算法来说，fit的过程其实就是保存原始数据集
        :param x: 输入特征空间
        :param y: 目标分类
        :return:
        """
        self.x = x
        self.y = y

    def __getClosest(self, x_pred, topK):
        """
        :desc 对于给定的数据，返回其对应的类
        :param x_pred: 输入的需要预测的数据
        :param topK: 指定的近邻个数
        :return: 对应的分类
        """
        # 首先计算该点到所有点的距离，将其存储在numpy列表之中
        distList = calDistance(self.x, x_pred)
        # 拿到最近的K个节点的索引
        index = distList.argsort()[:topK]
        # 拿到其对应的类别
        category = self.y[index]

        # 定义一个字典用于存储最近的K个类别中每个类别出现的个数
        labelList = dict(Counter([i for i in category]))
        label = None
        # 通过字典的方法找到出现最多的类别
        for k, v in labelList.items():
            if v == max(labelList.values()):
                label = k

        # 返回字典中value最大的key值
        return label

    def predict(self, x_pred, topK=10):
        """
        :desc 对于输入的数据进行预测
        :param x_pred: 需要预测的数据集
        :return: 返回预测的对应类别(np.array格式)
        """
        # 将输入的数据进行格式定义
        x_pred = x_pred.reshape(-1, self.x.shape[1])
        # 用于存储结果的类别
        categoryList = []
        # 对于需要预测的数据逐个的进行predict
        for data in x_pred:
            category = self.__getClosest(data, topK)
            categoryList.append(category)
        # 返回所有数据对应的类别
        return np.array(categoryList)


def calDistance(x, y):
    """
    :param x: 第一个矩阵
    :param y: 点
    :return: 两者之间的距离
    """
    diff = x - y
    distance = (diff * diff).sum(axis=1)
    Lp = np.sqrt(distance)
    return Lp


if __name__ == '__main__':
    # 导入数据
    mnist = datasets.fetch_openml('mnist_784', data_home='../mnist_dataset/')
    x = mnist['data'].values
    y = mnist['target'].values
    # 开始时间
    start = time.time()

    # 引入自己的对象
    KNN = MyKNN()
    # 将数据存入对象之中
    KNN.fit(x, y)
    # 随机选择需要预测的数据
    index = np.random.choice(x.shape[0])
    # 进行预测
    category = KNN.predict(x[index], 10)
    print(category)
    # plt.imshow(x[index].reshape(28,28))

    # 结束时间
    end = time.time()
    print('time span: ', end - start)
