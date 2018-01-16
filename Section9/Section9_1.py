from abupy import AbuFactorBuyBreak
from abupy import AbuFactorAtrNStop
from abupy import AbuFactorPreAtrNStop
from abupy import AbuFactorCloseAtrNStop
from abupy import abu
from abupy import AbuMetricsBase

if __name__ == '__main__':
    # 初始资金
    read_cash = 10000
    # 选股因子，这里选择不用
    stock_pickers = None
    # 买入因子，选用趋势突破中的向上突破
    buy_factors = [{'xd': 60, 'class': AbuFactorBuyBreak}, {'xd': 42, 'class': AbuFactorBuyBreak}]
    # 卖出因子，选择止盈止损，防暴跌，防盈利后亏损这三种因子
    sell_factors = [{'stop_loss_n': 1.0, 'stop_win_n': 3.0, 'class': AbuFactorAtrNStop},
                    {'class': AbuFactorPreAtrNStop, 'pre_atr_n': 1.5},
                    {'class': AbuFactorCloseAtrNStop, 'close_atr_n': 1.5}]
    # 股票池
    # choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG', 'usTSLA', 'usWUBA', 'usVIPS']
    choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU']
    # 运行策略
    abu_result_tuple, kp_pd_manager = abu.run_loop_back(read_cash, buy_factors, sell_factors, stock_pickers,
                                                        choice_symbols=choice_symbols, n_folds=2)
    # 对结果进行度量，打印度量结果
    metrics=AbuMetricsBase(*abu_result_tuple)
    metrics.fit_metrics()
    metrics.plot_returns_cmp()
    print(abu_result_tuple.head())
