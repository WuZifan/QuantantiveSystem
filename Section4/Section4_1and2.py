import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 创建初始数据
    view_days=504
    stock_cnt=200
    stock_day_change=np.random.standard_normal((stock_cnt,view_days))
    # print(stock_day_change[:2,:5])

    # 1、关于Dataframe的构建和相关方法
    stock_dataframe=pd.DataFrame(stock_day_change)
    # print(stock_dataframe.head())
    # 1.1 直接添加行索引
    stock_symbols=['Stock'+str(i) for i in range(0,stock_cnt)]
    stock_dataframe=pd.DataFrame(stock_day_change,index=stock_symbols)
    # print(stock_dataframe.head())
    # 1.2 直接添加列索引
    days=pd.date_range('2017-01-01',periods=view_days,freq='1D')
    stock_dataframe=pd.DataFrame(stock_day_change,index=stock_symbols,columns=days)
    # print(stock_dataframe.head())
    # 1.3.1 DataFrame的转置
    stock_dataframe=stock_dataframe.T
    # print(stock_dataframe)
    # 1.3.2 Dataframe的重采样
    stock_dataframe_resample=stock_dataframe.resample('21D').mean()
    # print(stock_dataframe_resample)
    # 1.4 Series,可以理解为只有一列的Dataframe
    stock_series=stock_dataframe['Stock0']
    # print(stock_series.head())
    # 顺手直接利用pandas直接画图
    stock_series.cumsum().plot()
    # plt.show()
    # 1.5 重采样数据
    stock_0_series=stock_dataframe['Stock0']
    # 这里ohlc指open，highest，lowest，close四个值
    # 分别对应5天累加结果中第一天，最高，最低和最后一天的值
    stock_resample5=stock_0_series.cumsum().resample('5D').ohlc()
    stock_resample20=stock_0_series.cumsum().resample('20D').ohlc()
    # 验证ohlc
    # print(stock_resample5.head())
    # print(stock_0_series.cumsum().head(20))
    # 用value得到series中的值后，就是一个numpy的array
    # print(type(stock_resample5['open'].values))

    # 2、基本数据分析
    from abupy import ABuSymbolPd
    # 获取特斯拉(TSLA),2年内数据
    tsla_df=ABuSymbolPd.make_kl_df('usTSLA',n_folds=2)
    print(tsla_df.tail())

    # 2.1.1 大致绘制股票走势，看一下样子
    # tsla_df[['close','volume']].plot(subplots=True,style=['r','g'],grid=True)
    # plt.show()
    # 2.1.2 利用info(),看一下数据类型
    print(tsla_df.info())
    # 2.1.3 利用describe(),看一下统计值
    print(tsla_df.describe())

    # 2.2.1 利用loc(),通过索引名称来进行切片,后面不传列值表明要所有的列
    print(tsla_df.loc['2017-01-01':'2017-01-10','close'])
    # 2.2.2 利用iloc，通过缩影的数值选取切片，这个就和python或者numpy一样
    print(tsla_df.iloc[1:10,2:4])
    # 2.2.3 还可以通过.列名的方式执行列
    print(tsla_df.open[0:3])
    # 2.2.4 混合使用iloc和loc,不过注意在获取多个列的时候，用两个[]
    print(tsla_df[['open','close','volume']][0:4])

    # 2.3.1 利用逻辑筛选找出符合要求的数据。
    #       注意，这里不像numpy一样只取出符号要求的那个数据，而是把符合数据所在一行都留下
    #       复合条件和numpy中一样
    print(tsla_df[(np.abs(tsla_df.p_change>8) & (tsla_df.volume>2.5*tsla_df.volume.mean()))])

    # 2.4.1 数据序列排序,通过by来选择按照哪一列，通过ascending来指定排序方向
    print(tsla_df.sort_index(by='p_change')[:5])
    # 2.4.2 处理丢失数据,丢弃drop，或者填充fill
    tsla_df=tsla_df.dropna(how='all')
    # inplace表示不用返回新的序列，就在原始序列上修改
    tsla_df.fillna(0,inplace=True)
    # 2.4.3 计算某一列的增长下跌幅度,但是没有把这列数据放到原来的dataframe中哦需要的话自己放
    print(tsla_df.close.pct_change()[:5])
    tsla_df['pct_change']=tsla_df.close.pct_change()
    print(tsla_df[:5])
