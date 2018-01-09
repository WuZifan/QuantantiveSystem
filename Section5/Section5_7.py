from scipy import stats
from abupy import ABuSymbolPd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    tsla_df=ABuSymbolPd.make_kl_df('TSLA',n_folds=2)
    # 统计上的黄金分割线
    sp382_stats=stats.scoreatpercentile(tsla_df.close,38.2)
    sp618_stats=stats.scoreatpercentile(tsla_df.close,61.8)
    # 视觉上的
    sp382_view=(tsla_df.close.max()-tsla_df.close.min())*0.382+tsla_df.close.min()
    sp618_view=(tsla_df.close.max()-tsla_df.close.min())*0.618+tsla_df.close.min()

    # 从视觉618和统计618中筛选更大的值
    above618 = np.maximum(sp618_view, sp618_stats)
    # 从视觉618和统计618中筛选更小的值
    below618 = np.minimum(sp618_view, sp618_stats)
    # 从视觉382和统计382中筛选更大的值
    above382 = np.maximum(sp382_view, sp382_stats)
    # 从视觉382和统计382中筛选更小的值
    below382 = np.minimum(sp382_view, sp382_stats)

    # 绘制收盘价
    plt.plot(tsla_df.close)
    # 水平线视觉382
    plt.axhline(sp382_view, c='r')
    # 水平线统计382
    plt.axhline(sp382_stats, c='m')
    # 水平线视觉618
    plt.axhline(sp618_view, c='g')
    # 水平线统计618
    plt.axhline(sp618_stats, c='k')

    # 填充618 red
    plt.fill_between(tsla_df.index, above618, below618,
                     alpha=0.5, color="r")
    # 填充382 green
    plt.fill_between(tsla_df.index, above382, below382,
                     alpha=0.5, color="g")

    plt.show()