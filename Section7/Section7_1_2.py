import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abupy import ABuSymbolPd


def plot_train_test(train_kl, test_kl):
    '''
    可视化训练数据和测试数据
    :param train_kl:
    :param test_kl:
    :return:
    '''
    fig, axis = plt.subplots(nrows=2, ncols=1)
    axis[0].plot(np.arange(0, train_kl.shape[0]), train_kl.close)
    axis[1].plot(np.arange(0, test_kl.shape[0]), test_kl.close)
    plt.show()

def plot_buy_sell_train_test(train_kl, test_kl):
    '''
    可视化训练数据，测试数据，买入水平线，卖出水平线
    :param train_kl:
    :param test_kl:
    :return:
    '''
    # 计算均值和方差
    train_close_mean = train_kl.close.mean()
    train_close_std = train_kl.close.std()

    # 构造买出信号阈值
    buy_signal = train_close_mean - (train_close_std / 3)
    # 构造卖出信号阈值
    sell_signal = train_close_mean + (train_close_std / 3)

    # 可视化
    train_kl.close.plot()
    # 画出买入信号
    plt.axhline(buy_signal, color='r', lw=3)
    # 画出卖出信号
    plt.axhline(sell_signal, color='g', lw=3)
    # 画出均值线
    plt.axhline(train_close_mean, color='black', lw=1)
    # 添加图例
    plt.legend(['train_close', 'buy_signal:' + str(buy_signal), 'sell_signal:' + str(sell_signal),
                'mean:' + str(train_close_mean)], loc='best')
    plt.title('Training')
    plt.show()

    # 在后一年数据上画图
    test_kl.close.plot()
    # 画出买入信号
    plt.axhline(buy_signal, color='r', lw=3)
    # 画出卖出信号
    plt.axhline(sell_signal, color='g', lw=3)
    # 画出均值线
    plt.axhline(train_close_mean, color='black', lw=1)
    # 添加图例
    plt.legend(['train_close', 'buy_signal:' + str(buy_signal), 'sell_signal:' + str(sell_signal),
                'mean:' + str(train_close_mean)], loc='best')
    plt.title('BackTest')
    plt.show()



if __name__ == '__main__':
    kl_pd = ABuSymbolPd.make_kl_df('AGYS', n_folds=2)
    # 1、得到训练数据测试数据
    train_kl = kl_pd[:252]
    test_kl = kl_pd[252:]

    # 2、计算均值和方差
    train_close_mean = train_kl.close.mean()
    train_close_std = train_kl.close.std()

    # 3、计算阈值
    # 构造买出信号阈值
    buy_signal = train_close_mean - (train_close_std / 3)
    # 构造卖出信号阈值
    sell_signal = train_close_mean + (train_close_std / 3)

    # 4、在测试集中设置买入卖出点
    # 寻找测试数据中满足买入条件的时间序列
    buy_index=test_kl[test_kl['close'] <= buy_signal].index
    # 把这些点设置为1，表示可以持有
    test_kl.loc[buy_index,'signal']=1
    # 寻找测试数据中满足卖出条件的时间序列
    sell_index=test_kl[test_kl.close >=sell_signal].index
    # 把这些点设置为1，表示该卖了
    test_kl.loc[sell_index,'signal']=0
    # 打印
    print(test_kl)

    # 5、处理数据
    # 另外，这里假设是全仓操作，一旦买了，就all in；一旦卖了，也就all out
    # 添加数据列 keep
    test_kl['keep']=test_kl['signal']
    # 令keep列中的缺失值，按照'ffill' 就是将缺失值按照前面一个值进行填充。
    test_kl['keep'].fillna(method='ffill',inplace=True)

    # 6、计算基准收益
    test_kl['benchmark_profit']=test_kl.close/test_kl.close.shift(1)-1

    # 7、计算持有股票的时候的收益
    test_kl['trend_profit']=test_kl.benchmark_profit*test_kl.keep

    # 注意，上面的计算值只是每天的收益，真正计算时，需要用累加值来算
    # 8、可视化，这个图告诉我们，它没有让利润奔跑啊
    fig,axis=plt.subplots(ncols=1,nrows=2)
    axis[0].plot(np.arange(0,test_kl.shape[0]),test_kl.close)
    axis[1].plot(np.arange(0,test_kl.shape[0]),test_kl.benchmark_profit.cumsum())
    axis[1].plot(np.arange(0,test_kl.shape[0]),test_kl.trend_profit.cumsum())
    axis[1].legend(['benchmark','trend_profit'])
    plt.show()



