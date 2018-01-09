import pandas as pd
import numpy as np
# 虽然用pandas也可以画图，但是由于pandas底层也是调用matplotlib的，所以在用的时候还是需要导入matplotlib的。
import matplotlib.pyplot as plt
from abupy import ABuSymbolPd

if __name__ == '__main__':
    # 2.1 演示什么叫做波动
    # 即，若干天的标准差乘以天数的开平方
    # demo_list=np.array([2,4,16,20])
    # demo_window=3
    # temp_result=pd.rolling_std(demo_list,window=demo_window,center=False)
    # rolling_result=temp_result*np.sqrt(demo_window)
    # print(rolling_result)

    # 3.1 使用pandas画波动曲线
    tsla_df=ABuSymbolPd.make_kl_df('TSLA',n_folds=2)
    tsla_df_copy=tsla_df.copy()
    # 这里不用像书里一样用shift，是因为数据中已经帮你填好了这个数据
    # 计算投资回报，即当天收盘价比上前天收盘价
    tsla_df_copy['return']=np.log(tsla_df['close']/tsla_df['pre_close'])
    # 移动收益标准差，计算20天收益率的标准差
    tsla_df_copy['mov_std']=pd.rolling_std(tsla_df_copy['return'],window=20,center=False)*np.sqrt(20)
    # 加权移动收益标准差
    tsla_df_copy['std_ewm']=pd.ewmstd(tsla_df_copy['return'],span=20,min_periods=20,adjust=True)*np.sqrt(20)
    # 画图
    # tsla_df_copy[['close','mov_std','std_ewm','return']].plot(subplots=True,grid=True)
    # plt.show()

    # 3.2 绘制股票的价格和均线
    # 价格
    # tsla_df.close.plot()
    # # ma30
    # pd.rolling_mean(tsla_df.close,window=30).plot()
    # plt.legend(['close','ma30'],loc='best')
    # plt.show()

    # 3.3 其他pandas统计图形种类
    # 获得低开高走的交易日的下一个交易日内容
    low_to_high_df=tsla_df.iloc[tsla_df[(tsla_df.close>tsla_df.open) & (tsla_df.key != tsla_df.shape[0]-1)].key.values+1]
    # 将涨跌幅，正的向上取整，负的向下取整
    # 这里指定了列，为p_change，因此np.where返回的时候，是一个n*1的列向量
    change_ceil_floor=np.where(low_to_high_df.p_change>=0,np.ceil(low_to_high_df['p_change']),np.floor(low_to_high_df['p_change']))
    # 由于上面返回的numpy数据，所有将其转化为pandas的series数据，方便绘图
    change_ceil_floor=pd.Series(change_ceil_floor)
    # 下面绘制不同类型的统计图
    _,axs=plt.subplots(nrows=2,ncols=2)
    # 竖直树状图
    change_ceil_floor.value_counts().plot(kind='bar',ax=axs[0][0])
    # 水平树状图
    change_ceil_floor.value_counts().plot(kind='barh',ax=axs[0][1])
    # 概率密度图
    # 说实话，我觉得这个概率密度图有问题，x的取值为什么能够那么大，不是应该集中在-4到4之间才对嘛
    # 这里貌似画反了
    change_ceil_floor.value_counts().plot(kind='kde',ax=axs[1][0])
    # 饼图
    change_ceil_floor.value_counts().plot(kind='pie',ax=axs[1][1])
    print(change_ceil_floor.value_counts())
    plt.show()
