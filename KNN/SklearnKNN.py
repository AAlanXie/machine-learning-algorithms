"""
coding: utf-8
author: tianqi
email: tianqixie98@gmail.com
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def score(pred, target):
    """
    :param pred: 预测数据
    :param target: 实际数据
    :return: 准确率
    """
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # 定义模型
    model = KNeighborsClassifier(n_neighbors=5)
    # 训练模型
    model.fit(x_train, y_train)
    # 预测数据
    y_pred = model.predict(x_test)
    # 拿到分数
    score = score(y_pred, y_test)
    # 打印分数
    print(f"accur: {score}")


