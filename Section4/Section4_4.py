import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abupy import ABuSymbolPd

if __name__ == '__main__':
    tsla_df=ABuSymbolPd.make_kl_df('TSLA',n_folds=2)
    tsla_df['positive']=np.where(tsla_df.p_change>0,1,0)
    print(tsla_df.head())

    # 构建交叉表，说白了就是某一列的所有取值做行，某一列的所有取值做列，画一个表
    # 表中每一个数字表示符合行和列条件的数据的个数。
    xt=pd.crosstab(tsla_df.date_week,tsla_df.positive)
    # 计算比例
    xt_div=xt.div(xt.sum(1).astype(float),axis=0)
    print(xt_div)

    # 还有一种透视表，貌似是只取了positive为1的情况
    # 感觉还是交叉表更加清晰
    pivot_tb=tsla_df.pivot_table(['positive'],index=['date_week'])
    print(pivot_tb)