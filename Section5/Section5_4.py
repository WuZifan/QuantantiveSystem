import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from abupy import ABuSymbolPd

if __name__ == '__main__':
    tsla_df=ABuSymbolPd.make_kl_df('TSLA',n_folds=2)
    print(tsla_df.head())
    # 一张图上画出直方图和概率密度图
    # sns.distplot(tsla_df['p_change'],bins=80)
    # plt.show()

    # 绘制箱体图
    # sns.boxplot(x=tsla_df['date_week'],y=tsla_df['p_change'],data=tsla_df)
    # plt.show()

    # 绘制相关性和概率分布图
    # sns.jointplot(tsla_df['high'],tsla_df['low'])
    # plt.show()

    # 绘制热力图，用来表示协方差矩阵
    change_df=pd.DataFrame({'tsla':tsla_df['p_change']})
    goog_df=ABuSymbolPd.make_kl_df('GOOG',n_folds=2)
    change_df=pd.concat([change_df,pd.DataFrame({'goog':goog_df['p_change']})],axis=1)
    aapl_df=ABuSymbolPd.make_kl_df('AAPL',n_folds=2)
    change_df=pd.concat([change_df,pd.DataFrame({'AAPL':goog_df['p_change']})],axis=1)
    print(change_df.head())
    # 计算协方差
    corr=change_df.corr()
    _,ax=plt.subplots()
    # 画出热力图
    sns.heatmap(corr,ax=ax)
    plt.show()