import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abupy import ABuSymbolPd

if __name__ == '__main__':
    tsla_df=ABuSymbolPd.make_kl_df('TSLA',n_folds=2)
    print(tsla_df.head())
    jump_threshold=tsla_df.close.mean()*0.03
    print(jump_threshold)

    # 创建一个新的Dataframe
    jump_df=pd.DataFrame()
    def judge_jump(today):
        # 如果你想要为一个定义在函数外的变量赋值，
        # 那么你就得告诉Python这个变量名不是局部的，而是 全局 的。
        # 我们使用global语句完成这一功能。
        # 没有global语句，是不可能为定义在函数外的变量赋值的。
        global jump_df,jump_threshold

        # 对于上涨的情况而言，满足下面条件才是跳空
        if today.p_change>0 and np.abs(today.low-today.pre_close)>jump_threshold:
            today['jump']=1
            today['jump_power']=np.abs(today.low-today.pre_close)/jump_threshold
            jump_df=jump_df.append(today)
        elif today.p_change<0 and np.abs(today.high-today.pre_close)>jump_threshold:
            today['jump']=-1
            today['jump_power']=np.abs(today.high-today.pre_close)/jump_threshold
            jump_df=jump_df.append(today)

    # 用for循环
    # for ind in np.arange(0,tsla_df.shape[0]):
    #     today=tsla_df.ix[ind]
    #     judge_jump(today)

    # for循环效率太低，可以用apply，这个和map类似，都是把函数作用在list的每一个数据上
    # 只不过apply是把函数作用在每一行数据上
    # tsla_df指要被执行函数的dataframe
    # 注意这里是axis=1，表示每次取数据时，沿着列的index遍历，那么就得到了一行数据
    tsla_df.apply(judge_jump,axis=1)
    print(jump_df.head())