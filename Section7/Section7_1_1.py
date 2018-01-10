from abupy import ABuSymbolPd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


if __name__ == '__main__':
    kl_pd=ABuSymbolPd.make_kl_df('TSLA',n_folds=2)
    sns.set_context(rc={'figure.figsize':(14,7)})
    # 简单的方法来显示趋势
    sns.regplot(x=np.arange(0,kl_pd.shape[0]),y=kl_pd.close.values,marker='+')
    plt.show()
