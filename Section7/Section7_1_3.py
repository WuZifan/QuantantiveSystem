import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abupy import ABuSymbolPd

if __name__ == '__main__':
    kl_pd=ABuSymbolPd.make_kl_df('TSLA',n_folds=2)
    # 1、这里采用N日趋势突破，即超过N1天内的最高价，就买入，低于N2天内的最低价，就卖出
    N1=42
    N2=21

    # 2.1 采用pd.rolling_max可以寻找一个窗口长度内最大值
    kl_pd['n1_high']=pd.rolling_max(kl_pd['high'],window=N1)
    # 2.2 但这样会导致前N1-1个元素为NAN，
    # 我们使用pd.expanding_max来填充NAN，expanding_max会逐个遍历数组元素，并把返回直到当前位置看到过的最大元素
    # 用前k天的收盘价来代替，k∈[0,N1]
    expan_max=pd.expanding_max(kl_pd['close'])
    kl_pd['n1_high'].fillna(expan_max,inplace=True)
    # 2.3 最小值同理
    kl_pd['n2_low']=pd.rolling_min(kl_pd['low'],window=N2)
    expan_min=pd.expanding_min(kl_pd['close'])
    kl_pd['n2_low'].fillna(expan_min,inplace=True)
    print(kl_pd.head())

    # 3.1 根据n1_high和n2_low来定义买入卖出的信号序列
    # 注意，是当天的收盘价，高于昨天以前的n1值，就买入，不能包括今天的，因为今天的收盘价，怎么也不会高于今天的最高值
    buy_signal=kl_pd[kl_pd.close > kl_pd.n1_high.shift(1)].index
    kl_pd.loc[buy_signal,'signal']=1
    # 3.2 n2_low的卖出信号同理
    sell_signal=kl_pd[kl_pd.close < kl_pd.n2_low.shift(1)].index
    kl_pd.loc[sell_signal,'signal']=0
    # 3.3 这里可以不用考虑Nan的情况
    kl_pd.signal.value_counts().plot(kind='pie')
    plt.show()

    # 4.1 将买入卖出的信号转化为持股的状态
    # 由于买入卖出的信号用了当天的收盘价，所以真正的持有和抛售是在第二天进行的
    # 所以这里要shifit
    kl_pd['keep']=kl_pd['signal'].shift(1)
    # 填充Nan
    kl_pd['keep'].fillna(method='ffill',inplace=True)
    kl_pd['keep'].fillna(0)
    print(kl_pd)

    # 5. 基准收益，是指我从一开始就持有，然后一直到最后才卖的收益
    # 5.1 计算每天基准收益
    kl_pd['benchmark_profit']=kl_pd['close']/kl_pd['close'].shift(1)-1
    # 5.2 计算策略每天收益
    kl_pd['trend_profit']=kl_pd['keep']*kl_pd['benchmark_profit']
    # 5.3 计算累加基准收益和策略收益
    kl_pd['benchmark_profit_accum']=kl_pd['benchmark_profit'].cumsum()
    kl_pd['trend_profit_accum']=kl_pd['trend_profit'].cumsum()


    print(kl_pd.head(10))

    # 5.4 可视化
    kl_pd[['benchmark_profit_accum','trend_profit_accum']].plot()
    plt.show()


