"""
author: tianqi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def OneDimensional(x, y):
    """
    :desc 对于一元线性回归来说，可以使用求偏导的方式直接求出对应的参数值，方程式由下列方式表现
    :param x: 输入特征空间
    :param y: 输出特征空间
    :return:
    """
    m = x.shape[0]
    # 对应w：权重 weight
    a = sum(y*(x - x.mean(axis=0))) / (sum(x * x) - (sum(x) * sum(x)) / m)
    # 对应b：偏差 bias
    b = sum(y - a*x) / m
    y_pred = a * x + b
    return y_pred

if __name__ == '__main__':
    # 构造随机数据
    x = np.random.randint(1, 100, size=(50, 1), dtype='int32')
    y = x * 0.2 + 10 + np.random.rand(50, 1) * 3
    # 使用自己的方程式解决
    y_pred = OneDimensional(x, y)

    # 调用sklearn包中的对应类解决
    # model = LinearRegression()
    # model.fit(x, y)
    # y_pred = model.predict(x)

    index = np.argsort(x, axis=0).reshape(1, -1)
    plt.plot(np.sort(x, axis=0), y_pred[index].reshape(-1, 1), 'red')
    plt.show()




