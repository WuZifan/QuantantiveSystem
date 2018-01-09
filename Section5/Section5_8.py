from scipy import stats
from abupy import ABuSymbolPd
import numpy as np
import matplotlib.pyplot as plt
from abupy import nd

if __name__ == '__main__':
    tsla_df=ABuSymbolPd.make_kl_df('TSLA',n_folds=2)

    nd.macd.plot_macd_from_klpd(tsla_df)