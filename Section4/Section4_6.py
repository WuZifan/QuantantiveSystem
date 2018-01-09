import numpy as np
import pandas as pd
from abupy import ABuIndustries

if __name__ == '__main__':
    r_symbol='usTSLA'
    # 获取和特斯拉同一个行业的股票，构成一个k*n*m的面板数据
    # 可以理解为k个股票，每个股票的dataframe和前面一样
    p_date,_=ABuIndustries.get_industries_panel_from_target(r_symbol,show=False)
    # Dimensions: 7 (items) x 499 (major_axis) x 12 (minor_axis)
    # 查看数据,发现是7*499*12，表示7支股票，499天，12个特征
    # print(p_date)
    # 沿着items方向进行切分
    print(p_date['usTTM'].head())
    # 但是无法沿着major_axis或者minor_axis方向气氛，
    # 比如我想所有股票每一天的close价格，那么是要沿着close切分
    # print(p_date['close'].head()) 这么写会报错

    # 如果想要沿着close切分，那么首先要把close所在的轴(这里是minor_axis)变成items轴
    p_data_it=p_date.swapaxes('items','minor')
    print(p_data_it['close'].head())