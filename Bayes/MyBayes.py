"""
coding: utf-8
author: tianqi
email: tianqixie98@gmail.com
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

class MyBayes(object):
    def __init__(self):
        self.x, self.y = None, None
        self.sample_numbers = None
        self.feature_numbers = None

    def fit(self, x, y):
        # 将x和y录入
        self.x, self.y = x, y
        self.sample_numbers, self.feature_numbers = x.shape[0], x.shape[1]

    def __getCategory(self, x_pred):
        """
        :param x_pred: 输入的需要预测的数据
        :return: 返回对应分类
        :desc
        参考: P(X|Y) = P(Y|X) * P(X) / P(Y)
        1. 首先计算出先验概率分布
            category_ratio 即对于每一种target的类别出现的概率
        2. 计算条件概率分布
            feature_ratio 对于每一条预测数据来说，分析其每个特征在对应target类别中出现的概率，将其累乘即为本数据出现在target类别中的概率
        3. 将两者相乘即为最终的概率
        其中会出现累乘项为0或者最终结果过于小计算机精度无法达到的情况，会使用一些方法将其进行展现(取log，取后乘法变为加法)
        """
        # 用于记录所有类别对应的相关概率信息
        category_list = dict()
        # 统计类别的个数
        k1 = len(set(self.y))
        for category in set(self.y):
            # 拿到类别中所有行的索引
            index = np.array(self.y == category)
            # 统计总共有多少条数据属于本类别并计算本类别占比
            category_ratio = math.log((np.sum(index) + 1.0) / (self.sample_numbers + k1))
            # 拿出所有属于本类别的数据
            ck = self.x[index, :]
            # 预先定义特征的相关概率用于累乘
            feature_ratio = 0
            # 对x_pred中的每一个特征进行分析
            for feature_index in range(x_pred.shape[0]):
                # 拿到对应特征的那一列
                x_i = ck[:, feature_index]
                # 统计对应的特征中有几种分类
                s1 = len(set(x_i))
                # 计算那一列中有多少数据和预测特征相同
                idx = np.array(x_i == x_pred[feature_index])
                # 拿到对应特征的相关概率
                feature_ratio = feature_ratio + math.log((np.sum(idx) + 1.0) / (np.sum(index) + s1))
            # 字典中写入相关的概率
            category_list[category] = category_ratio + feature_ratio

        final_category = None
        for key, value in category_list.items():
            if value == max(category_list.values()):
                # 找到最大的类别并赋值
                final_category = key

        # 返回最后的分类情况
        return final_category

    def predict(self, x_pred):
        # 将输入的数据进行格式定义
        x_pred = x_pred.reshape(-1, self.x.shape[1])
        # 用于存储结果的类别
        category_list = []
        # 记录进度
        number = 0
        # 对于需要预测的数据逐个的进行predict
        for data in x_pred:
            category = self.__getCategory(data)
            category_list.append(category)
            # 打印进度
            number = number + 1
            print(f"{number * 1.0 / x_pred.shape[0]} has been finished !! ")
        # 返回所有数据对应的类别
        return np.array(category_list)

def score(pred, target):
    # 定义一个值用来存储预测失败的数量
    error = 0
    # 对比每个值来判断准确性
    for i in range(len(pred)):
        if pred[i] != target[i]:
            error += 1
    return (len(pred) - error) * 1.0 / len(pred)

if __name__ == '__main__':
    # 导入数据
    mnist = datasets.fetch_openml('mnist_784', data_home='../mnist_dataset/')
    x = mnist['data'].values
    y = mnist['target'].values
    # 将其划分为训练数据集和测试数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=0)

    # 引入自己的模型
    model = MyBayes()
    # 引入训练数据集
    model.fit(x_train, y_train)
    # 预测数据
    y_pred = model.predict(x[0:10])
    # 准确率
    score = score(y_pred, y[0:10])
    # 打印准确率
    print(f'accur: {score}')



