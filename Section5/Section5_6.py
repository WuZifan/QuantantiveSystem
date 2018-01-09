import numpy as np
import matplotlib.pyplot as plt
from abupy import ABuSymbolPd

def regular_std(group):
    '''
    归一化，令起均值为0，标准差为1
    :param group:
    :return:
    '''
    return (group-group.mean())/group.std()

def regular_min_max(group):
    '''
    另一种归一化，减去最小值，除以（最大值减去最小值）
    :param group:
    :return:
    '''
    return (group-group.min())/(group.max()-group.min())

def two_mean_list(one,two,type_look='look_max'):
    '''
    向较大序列看齐：较小序列乘以（较大序列的均值除以较小序列的均值）
    向较小序列看同理
    :param one: 第一个序列
    :param two: 第二个序列
    :param type_look: 看齐方式
    :return:
    '''
    one_mean = one.mean()
    two_mean = two.mean()
    if type_look == 'look_max':
        """
            向较大的均值序列看齐
        """
        one, two = (one, one_mean / two_mean * two) \
            if one_mean > two_mean else (
            one * two_mean / one_mean, two)
    elif type_look == 'look_min':
        """
            向较小的均值序列看齐
        """
        one, two = (one * two_mean / one_mean, two) \
            if one_mean > two_mean else (
            one, two * one_mean / two_mean)

    return one,two

def plot_two(drawer,one,two,title):
    drawer.plot(one)
    drawer.plot(two)
    drawer.set_title(title)
    drawer.legend(['TSLA','GOOG'],loc='best')

if __name__ == '__main__':
    tsla_df=ABuSymbolPd.make_kl_df('TSLA',n_folds=2)
    goog_df=ABuSymbolPd.make_kl_df('GOOG',n_folds=2)


    # 1.为了更好的比较趋势，需要把数据进行归一化
    _,axs=plt.subplots(nrows=2,ncols=2)
    # 第一种归一化
    plot_two(axs[0][0],regular_std(tsla_df.close),regular_std(goog_df.close),'regular_std')

    # 第二种归一化
    plot_two(axs[0][1], regular_min_max(tsla_df.close), regular_min_max(goog_df.close), 'regular_min_max')

    # 第三种归一化
    tsla_df_re,goog_df_re=two_mean_list(tsla_df.close,goog_df.close)
    plot_two(axs[1][0], tsla_df_re, goog_df_re, 'look_max')

    # 第四种归一化
    tsla_df_re,goog_df_re=two_mean_list(tsla_df.close,goog_df.close,type_look='look_min')
    plot_two(axs[1][1], tsla_df_re, goog_df_re, 'look_min')

    plt.show()

    # 2.上面的归一化是一种方法，还可以使用双y轴策略
    _,ax1=plt.subplots()
    ax1.plot(tsla_df.close,c='r',label='tsla')
    ax1.legend(loc=2)
    ax1.grid(False)
    # 这里是关键，相当于在同一个图上，两个x轴重合
    ax2=ax1.twinx()
    ax2.plot(goog_df.close,c='g',label='google')
    ax2.legend(loc=1)

    plt.show()