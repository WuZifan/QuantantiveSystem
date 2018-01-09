import pandas as pd
import numpy as np
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
from abupy import ABuSymbolPd

if __name__ == '__main__':
    # 1.1 下面演示的是如何使用sklearn来进行线性拟合
    tsla_df=ABuSymbolPd.make_kl_df('TSLA',n_folds=2)
    print(tsla_df.head())
    # 准备拟合数据，x必须是dataframe类型，y必须是series类型
    y=tsla_df.close
    x=(np.arange(0,tsla_df.shape[0]))
    x=pd.DataFrame(x)
    # 调用sklearn的线性拟合模块开始拟合，并且需要计算截距
    model=LinearRegression(fit_intercept=True)
    model.fit(x,y)
    # 打印截距和参数
    print(model.intercept_)
    print(model.coef_)
    # 根据上面两个参数，得到预测直线的y值，或者用公式算也可以
    y_pred=model.predict(x)
    # y_pred=model.coef_*x+model.intercept_
    # 计算评价指标
    MAE=metrics.mean_absolute_error(y,y_pred)
    MSE=metrics.mean_squared_error(y,y_pred)
    RMSE=np.sqrt(MSE)
    # 打印评价指标
    print(MAE)
    print(MSE)
    print(RMSE)
    # 画图查看
    plt.plot(np.arange(0,tsla_df.shape[0]),tsla_df.close)
    plt.plot(np.arange(0,tsla_df.shape[0]),y_pred)
    plt.show()

    # 1.2 当然sklearn也是可以使用多项式回归的
    # 首先重新写一下x和y值
    x = (np.arange(0, tsla_df.shape[0]))
    x = pd.DataFrame(x)
    y = tsla_df.close

    # 我们画四张图
    fig,axs=plt.subplots(ncols=2,nrows=2)
    # 把axs转化成list
    axs_list=list(itertools.chain.from_iterable(axs))
    # 我们假设我们从1~4次多项式回归
    model=LinearRegression(fit_intercept=True)
    for i in range(0,4):
        # 如何实现多项式拟合
        pf=PolynomialFeatures(degree=i+1)
        model.fit(pf.fit_transform(x),y)
        # 得到预测值
        y_pred=model.predict(pf.fit_transform(x))
        axs_list[i].plot(x,y)
        axs_list[i].plot(x,y_pred)

    plt.show()

