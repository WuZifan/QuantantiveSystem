import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abupy import AbuPickRegressAngMinMax
from abupy import AbuPickStockWorker
from abupy import AbuCapital
from abupy import AbuKLManager
from abupy import AbuBenchmark
from abupy import ABuRegUtil
from abupy import ABuPickStockExecute
from abupy import AbuPickStockPriceMinMax

if __name__ == '__main__':
    '''
        1)
        这里是来进行选股策略的编写。
        1.我们希望，对一段时间内股票的收盘价做线性拟合，得到拟合直线的斜率。
        2.当斜率大于我们定于的最小阈值，且小于最大阈值时，表明可以买入
    '''
    # 这里是条件，斜率最小值为0.0，即要求股票是上涨的趋势
    stock_pickers=[{'class':AbuPickRegressAngMinMax,'threshold_ang_min':0.0,'received':False}]

    # 一般而言，我们是遍历整个股市来选股，这里我们就选择以下几个股票来做演示
    choice_symbols=['usNOAH','usSFUN','usBIDU','usAAPL','usGOOG','usTSLA','usWUBA','usVIPS']

    # 开始执行
    benchmark = AbuBenchmark()
    capital=AbuCapital(1000000,benchmark)
    kl_pd_manager=AbuKLManager(benchmark,capital)
    stock_pick=AbuPickStockWorker(capital,benchmark,kl_pd_manager,choice_symbols=choice_symbols,stock_pickers=stock_pickers)
    stock_pick.fit()

    print(stock_pick.choice_symbols)

    # 绘图
    kl_pd_SFUN=kl_pd_manager.get_pick_stock_kl_pd('usNOAH')
    deg=ABuRegUtil.calc_regress_deg(kl_pd_SFUN.close)
    print(deg)

    # 上面使用worker的操作太麻烦，下面可以直接使用executer
    stock_pickers=[{'class':AbuPickRegressAngMinMax,'threshold_ang_min':0.0,'threshold_ang_max':10.0,'reversed':False}]
    result=ABuPickStockExecute.do_pick_stock_work(choice_symbols,benchmark,capital,stock_pickers)
    print(result)


    '''
        2)
        同样，上面是采用了一种选股因子，那么，当我采用多种选股因子时会怎样呢？
        1. 这里注意，选股和择时是不一样的。
        2. 对于择时而言，当我有多因子时，任意时刻，只要满足择时中的一个条件，就可以进行买入或者卖出操作
        3. 而对于选股，只有当股票的趋势满足所有因子时，我们才选择该股票
    '''
    stock_pickers=[{'class':AbuPickRegressAngMinMax,'threshold_ang_min':0.0,'reversed':False},
                   {'class':AbuPickStockPriceMinMax,'threshold_price_min':50.0,'reversed':False}]
    result=ABuPickStockExecute.do_pick_stock_work(choice_symbols,benchmark,capital,stock_pickers)
    print(result)