import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abupy import ABuSymbolPd

def plot_series(tsla_series,alpha=1):
    plt.plot(tsla_series,alpha=alpha)


def plot_trade(tsla_df,start_date,end_date):
    '''
    标出持有期，当最后一天下跌了，区间为绿色（止损），最后一天上涨了，区间为红色（止盈）
    :param tsla_df: 股票数据
    :param start_date: 持有股票的那一天
    :param end_date: 卖出股票的那一天
    :return:
    '''
    # 得到购买股票的那一天
    start = tsla_df[tsla_df.index == start_date].key.values[0]
    # 找出2014-09-05对应时间序列中的index作为end
    end = tsla_df[tsla_df.index == end_date].key.values[0]

    # 使用5.1.1封装的绘制tsla收盘价格时间序列函数plot_demo
    # just_series＝True, 即只绘制一条曲线使用series数据
    plot_series(tsla_df.close)

    # 将整个时间序列都填充一个底色blue，注意透明度alpha=0.08是为了
    # 之后标注其他区间透明度高于0.08就可以清楚显示
    plt.fill_between(tsla_df.index, 0, tsla_df['close'], color='blue',
                     alpha=.08)

    # 标注股票持有周期绿色，使用start和end切片周期
    # 透明度alpha=0.38 > 0.08
    print(tsla_df[end:end+1])
    if tsla_df['close'][end] < tsla_df['open'][end]:
        plt.fill_between(tsla_df.index[start:end], 0,
                         tsla_df['close'][start:end], color='green',
                         alpha=.38)
    else:

        plt.fill_between(tsla_df.index[start:end], 0,
                         tsla_df['close'][start:end], color='red',
                         alpha=.38)

    # 设置y轴的显示范围，如果不设置ylim，将从0开始作为起点显示，效果不好
    plt.ylim(np.min(tsla_df['close']) - 5,
             np.max(tsla_df['close']) + 5)
    # 使用loc='best'
    plt.legend(['close'], loc='best')


def plot_trade_with_annotation(tsla_df,start,end,annotate):
    '''
    在绘图的基础上添加注解
    :param tsla_df:
    :param start:
    :param end:
    :param annotate: 注解
    :return:
    '''
    # 标注交易区间buy_date到sell_date
    plot_trade(tsla_df,start, end)
    # annotate文字，asof：从tsla_df['close']中找到index:sell_date对应值
    plt.annotate(annotate,
                 xy=(end, tsla_df['close'].asof(end)),
                 arrowprops=dict(facecolor='yellow'),
                 horizontalalignment='left', verticalalignment='top')
    plt.show()


if __name__ == '__main__':
    tsla_df=ABuSymbolPd.make_kl_df('TSLA',n_folds=2)
    print(tsla_df.head())

    plot_trade_with_annotation(tsla_df,'2016-07-28','2016-10-11','sell for gain money')
