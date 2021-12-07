"""
coding: utf-8
author: tianqi
email: tianqixie98@gmail.com
"""
from sklearn import datasets
# linear_model中有比较多的线性回归方法
"""
    LinearRegression: 
    如下两个方法课减少过拟合
    例如高矮胖瘦变成了四个参数，但其相关性极高，就会导致四个参数的可解释性减弱，加入新的惩罚项来将其减小
    Ridge: 岭回归，加了L2正则
    Lasso: 加了L1正则
    相同: 都可以解决普通线性回归的过拟合问题
    不同: lasso可以用来feature selection，L1项的正则可以使一些参数项为0
"""
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    bias = 100

    X = np.arange(1000).reshape(-1,1)
    y_true = np.ravel(X.dot(0.3) + bias)
    noise = np.random.normal(0, 60, 1000)
    y = y_true + noise

    # 拟合之中的fit_intercept参数是用来设置截距是否为0
    lr_fi_true = LinearRegression(fit_intercept=True)
    lr_fi_false = LinearRegression(fit_intercept=False)

    lr_fi_true.fit(X, y)
    lr_fi_false.fit(X, y)

    print('Intercept when fit_intercept=True : {:.5f}'.format(lr_fi_true.intercept_))
    print('Intercept when fit_intercept=False : {:.5f}'.format(lr_fi_false.intercept_))

    lr_fi_true_yhat = np.dot(X, lr_fi_true.coef_) + lr_fi_true.intercept_
    lr_fi_false_yhat = np.dot(X, lr_fi_false.coef_) + lr_fi_false.intercept_

    # 画出图像
    plt.scatter(X, y, label='Actual points')
    plt.plot(X, lr_fi_true_yhat, 'r--', label='fit_intercept=True')
    plt.plot(X, lr_fi_false_yhat, 'r-', label='fit_intercept=False')
    plt.legend()

    plt.vlines(0, 0, y.max())
    plt.hlines(bias, X.min(), X.max())
    plt.hlines(0, X.min(), X.max())

    plt.show()