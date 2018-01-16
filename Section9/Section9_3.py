import numpy as np
from abupy import AbuFactorBuyBreak
from abupy import AbuFactorAtrNStop
from abupy import AbuFactorPreAtrNStop
from abupy import AbuFactorCloseAtrNStop
from abupy import abu
from abupy import AbuMetricsBase
from abupy import ABuGridHelper
from abupy import GridSearch

if __name__ == '__main__':
    '''
        使用grid-search来寻找最优参数
    '''

    # 1.考虑不同卖出因子的组合
    # 设置止盈止损最优参数范围
    # stop_win_range = np.arange(2.0, 4.5, 0.5)
    # stop_loss_range = np.arange(0.5, 2, 0.5)
    stop_win_range = np.arange(2.0, 4, 0.5)
    stop_loss_range = np.arange(1, 2, 0.5)
    # 查看数据
    print("止盈%s" % stop_win_range)
    print("止损%s" % stop_loss_range)

    sell_atr_nstop_factor_grid = {'class': [AbuFactorAtrNStop], 'stop_loss_n': stop_loss_range,
                                  'stop_win_n': stop_win_range}

    # 设置防暴跌，防盈利后亏损参数
    # close_atr_range = np.arange(1.0, 4.0, 0.5)
    # pre_atr_range = np.arange(1.0, 3.5, 0.5)
    close_atr_range = np.arange(1.0, 3.0, 0.5)
    pre_atr_range = np.arange(1.0, 3, 0.5)

    print("close: %s" % close_atr_range)
    print("pre: %s" % pre_atr_range)

    sell_atr_pre_factor_grid = {'class': [AbuFactorPreAtrNStop], 'pre_atr_n': pre_atr_range}
    sell_atr_close_factor_grid = {'class': [AbuFactorCloseAtrNStop], 'close_atr_n': close_atr_range}

    # 对各个因子做排列组合
    sell_factors_product = ABuGridHelper.gen_factor_grid(ABuGridHelper.K_GEN_FACTOR_PARAMS_SELL,
                                                         [sell_atr_nstop_factor_grid, sell_atr_pre_factor_grid,
                                                          sell_atr_close_factor_grid])
    # 一共有476种方式，计算方法如下：
    #   1.全部组合，有5*3*6*5=450种
    #   2.仅考虑止盈止损，有5*3=15种
    #   3.仅考虑防暴跌：有6种
    #   4.仅考虑防盈利后亏损，有5种
    #   5.综上，一共476种
    print("卖出组合总的组合方式有%s" % len(sell_factors_product))
    print("第0中组合为%s" % sell_factors_product[0])

    # 2、再考虑买入因子
    buy_bk_factor_grid1 = {
        'class': [AbuFactorBuyBreak],
        'xd': [42]
    }

    buy_bk_factor_grid2 = {
        'class': [AbuFactorBuyBreak],
        'xd': [60]
    }

    buy_factors_product = ABuGridHelper.gen_factor_grid(
        ABuGridHelper.K_GEN_FACTOR_PARAMS_BUY, [buy_bk_factor_grid1, buy_bk_factor_grid2])

    # 有三种组合，分别为，只使用42日突破，只使用60日突破和同时使用42日和60日突破。
    print('买入因子参数共有{}种组合方式'.format(len(buy_factors_product)))
    print('买入因子组合形式为{}'.format(buy_factors_product))

    # 3、进行gridsearch
    read_cash = 100000
    choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU']
    grid_search = GridSearch(read_cash, choice_symbols, buy_factors_product=buy_factors_product,
                             sell_factors_product=sell_factors_product)
    scores,score_tuple_array=grid_search.fit(n_jobs=-1)

    print('组合因子参数数量{}'.format(len(buy_factors_product) * len(sell_factors_product)))
    # 如果从本地序列文件中读取则没有scores
    print('最终评分结果数量{}'.format(len(scores)))

    # 可视化策略
    best_score_tuple_grid = grid_search.best_score_tuple_grid
    AbuMetricsBase.show_general(best_score_tuple_grid.orders_pd, best_score_tuple_grid.action_pd,
                                best_score_tuple_grid.capital, best_score_tuple_grid.benchmark)
