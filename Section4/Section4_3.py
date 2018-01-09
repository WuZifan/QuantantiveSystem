import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abupy import ABuSymbolPd

if __name__ == '__main__':
    tsla_df=ABuSymbolPd.make_kl_df('TSLA',n_folds=2)
    # 画直方图感受一下
    # tsla_df.p_change.hist(bins=80)
    # plt.show()
    # 通过pd.qcut将p_change的取值尽可能平均划分到10个不相交的范围里，
    # 并通过value_counts来进行打印显示
    cats=pd.qcut(np.abs(tsla_df.p_change),10)
    print(cats.value_counts())

    # 上面是它帮我们定切分范围，如果自己定切分范围呢
    bins=[-np.inf,0,1,2,3,np.inf]
    cats=pd.cut(np.abs(tsla_df.p_change),bins)
    print(cats.value_counts())
    # 结合get_dummies，形成类似one-hot编码.
    # 用来查看任何一天特斯拉的涨跌幅，落在哪个范围内
    change_dummies=pd.get_dummies(cats,prefix='p_change_dummies')
    print(change_dummies[:5])

    # 把之前的得到的数据加入到原有数据中，如果只有一列的话，可以用这个
    # tsla_df['列index名']=要被加入的dataframe
    # 很多列的话，可以用concat,注意把要连在一起的数据写到[]里面
    tsla_df=pd.concat([tsla_df,change_dummies],axis=1)
    # pd.concat是会返回新的dataframe的，所以要用新的dataframe接住它才能够更新
    print(tsla_df)
    # concat，且axis=1时，是往dataframe加入更多的列
    # 当然，可以利用pd.appen加入更多的行
