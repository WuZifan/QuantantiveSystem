from abupy import AbuFactorBuyBase
from abupy import AbuBenchmark
from abupy import AbuPickTimeWorker
from abupy import AbuCapital
from abupy import AbuKLManager
from abupy import AbuFactorBuyBreak
from abupy import ABuTradeProxy
from abupy import ABuTradeExecute
from abupy import AbuFactorSellBreak
from abupy import ABuPickTimeExecute
from abupy import AbuFactorAtrNStop
from abupy import AbuFactorPreAtrNStop
from abupy import AbuFactorCloseAtrNStop
from abupy import AbuSlippageBuyBase
from abupy import AbuMetricsBase
from abupy import AbuKellyPosition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# class AbuFactorBuyBreak(AbuFactorBuyBase):
#     '''
#     实现N日趋势突破:
#         1.当第x天收盘后，发现第x天的股价最大值，大于前k天（不包括第x天）的最大值，那么在第x+1天买入股票
#         2.当第x天收盘后，发现第x天的股票最小值，小于前k天（不包括第x天）的最小值，那么在第x+1天卖出股票
#         3.在第x+1天买入股票后，从x+1天到x+1+k天这段时间，哪怕有新的最高值创出，也不再购入
#     '''
#     def _init_self(self, **kwargs):
#         # 突破参数xd，比如20天，30天突破
#         self.xd=kwargs['xd']
#         # 忽略连续创新高，比如买入第二天有创新高
#         self.skip_days=0
#         # 显示名字
#         self.factor_name='{}:{}'.format(self.__class__.__name__,self.xd)
#
#     def fit_day(self, today):
#         day_ind=int(today.key)
#         # 忽略不符合的买入日期（前xd天和最后一天）
#         if day_ind<self.xd-1 or day_ind>=self.kl_pd.shape[0]-1:
#             return None
#
#         if self.skip_days>0:
#             # 执行买入订单忽略
#             self.skip_days -= 1
#             return None
#
#         if today.close == self.kl_pd.close[day_ind-self.xd+1:day_ind+1].max():
#             # 买入股票后，在下一个xd天内，如果股票再创新高，也不买了
#             self.skip_days=self.xd
#             # 生成买入订单
#             return self.make_buy_order(day_ind)
#         return None

if __name__ == '__main__':
    '''
        1）
        实现N日趋势突破:
         1.当第x天收盘后，发现第x天的股价最大值，大于前k天（不包括第x天）的最大值，那么在第x+1天买入股票
         2.当第x天收盘后，发现第x天的股票最小值，小于前k天（不包括第x天）的最小值，那么在第x+1天卖出股票
         3.在第x+1天买入股票后，从x+1天到x+1+k天这段时间，哪怕有新的最高值创出，也不再购入

    '''
    # 1、这里是买入突破
    # 创建60日向上突破，42日向上突破两个因子
    buy_factors = [{'xd': 60, 'class': AbuFactorBuyBreak}, {'xd': 42, 'class': AbuFactorBuyBreak}]
    # 基准利润
    benchmark = AbuBenchmark()

    # 现金和基准利润
    capital = AbuCapital(1000000, benchmark)
    # 多线程管理类
    kl_pd_manager = AbuKLManager(benchmark, capital)
    # 获取TSLA的股票信息
    kl_pd = kl_pd_manager.get_pick_time_kl_pd('usTSLA')
    # 准备开始工作
    abu_worker = AbuPickTimeWorker(capital, kl_pd, benchmark, buy_factors, None)
    abu_worker.fit()
    # 画出哪几个点可以买入，以及最终的收益情况
    # orders_pd,action_pd,_=ABuTradeProxy.trade_summary(abu_worker.orders,kl_pd,draw=True)
    orders_pd, action_pd, _ = ABuTradeProxy.trade_summary(abu_worker.orders, kl_pd, draw=False)

    # 上面是从股价角度，下面从我们的资金来看
    # ABuTradeExecute.apply_action_to_capital(capital, action_pd, kl_pd_manager)
    # print(capital.capital_pd.head())
    # capital.capital_pd.capital_blance.plot()
    # plt.show()

    # 上面只是实现了买入突破，下面我们用卖出突破来试试
    # 2、这里是卖出突破
    sell_factor1 = {'xd': 120, 'class': AbuFactorSellBreak}
    sell_factors = [sell_factor1]
    capital = AbuCapital(1000000, benchmark)
    orders_pd, action_pd, _ = ABuPickTimeExecute.do_symbols_with_same_factors(['usTSLA'],
                                                                              benchmark, buy_factors, sell_factors,
                                                                              capital, show=False)

    '''
        下面会用到ATR这个指标，解释一下这个指标。
            1. 设当前交易日的最高价减最低价为x1，
            2. 前一个交易日的收盘件减去当前交易日的最高价的绝对值为x2
            3. 前一个交易日的收盘件减去当前交易日的最低价的绝对值为x3
            4. 那么真实波动TR=max(x1,x2,x3)
            5. ATR,就是求N天这个值（TR）的平均，ATR=(∑TR）/n
        作用：
            1. 用来衡量价格的波动强度，和价格的绝对值没有关系
    '''

    '''
        2）
        上面给出了买入点和卖出点的定义，以及他们在历史数据上的表现
        但是现在很多操作会有止盈和止损的操作，我们使用ATR来协助判断止盈和止损
        具体的止盈止损策略：
            1. 当添加了止盈策略时，我们选择atr21和atr14的和作为atr常数，人工传入一个止盈率win_rate
            2. 当当前的收益大于0，且大于止盈率乘以atr常数时，可以卖出股票了。
            3. 亏损同理，即atr常数相同，换一个止损率就好
    '''
    # 构造信号因子
    sell_factor2 = {'stop_loss_n': 0.5, 'stop_win_n': 3.0, 'class': AbuFactorAtrNStop}
    # 多因子同时生效
    sell_factors = [sell_factor1, sell_factor2]
    # 执行
    capital = AbuCapital(1000000, benchmark)
    orders_pd, action_pd, _ = ABuPickTimeExecute.do_symbols_with_same_factors(['usTSLA'], benchmark, buy_factors,
                                                                              sell_factors, capital, show=False)

    '''
        3）
        利用atr来进行止损有一个弊端。
        1.对于atr而言，我们认为，从买入股票到今天，所有的亏损值，如果大于atr常数乘以止损率，那么就卖出
        2.而如果在一天内，股价暴跌，由于atr的平均性质，它无法很快的反应出来，这样会导致我们无法及时出逃。
        3.(当然，atr这样也可以帮助我们不会因为一时的起伏就错过一只长远看有利可图的股票，这个就看自己选择了）
        4.那么，在生存第一的情况下，我们在暴跌发生时，马上逃离。

        或者这么理解
        1.在之前止损的时候，我们算的是总的亏损超过某个值，我们就逃。
        2.现在的情况是，在第n天的时候，我有利润，但没过止盈点，第n+1天的时候，股价暴跌，但还没有跌破我的止损点。
        3.那么，这一天之内发生的暴跌，是否应该作为我出逃股票的信号？
        4.一般认为是的，所以要把这个信息加入进入。
    '''
    # 构造因子
    sell_factor3 = {'class': AbuFactorPreAtrNStop, 'pre_atr_n': 1.0}
    # 多因子同时生效
    sell_factors = [sell_factor1, sell_factor2, sell_factor3]
    # 执行
    capital = AbuCapital(1000000, benchmark)
    orders_pd, action_pd, _ = ABuPickTimeExecute.do_symbols_with_same_factors(['usTSLA'], benchmark, buy_factors,
                                                                              sell_factors, capital, show=False)

    '''
        4）
        在之前的止盈策略中，我们是令全局的收益高过某一个动态值【（atr21+atr14)*rate】，才会卖出
        那么，在股票已经盈利，但是没有超过这个止盈点，而后续股价下跌，使得我们不得不止损卖出，是不是会觉得很亏
        即，本来可以卖出盈利，但是因为目标定的太高，没有实现，反倒使得自己亏损了。
        那么，我们就需要采用一定策略，来避免这样的情况。
        具体做法是：
            1.对于已经盈利，但没有超过止盈点的股票，我们设置一个保护止盈点。
            2.当盈利股票下跌超过保护止盈点后，我们就将其卖出。
    '''
    # 构造因子
    sell_factor4 = {'class': AbuFactorCloseAtrNStop, 'close_atr_n': 1.5}
    # 多因子同时生效
    sell_factors = [sell_factor1, sell_factor2, sell_factor3, sell_factor4]
    # 执行
    capital = AbuCapital(1000000, benchmark)
    orders_pd, action_pd, _ = ABuPickTimeExecute.do_symbols_with_same_factors(['usTSLA'], benchmark, buy_factors,
                                                                              sell_factors, capital, show=False)

    '''
        5）
        1.上面的交易策略大致的总结是这样，我们在第x天交易结束后，往回看，根据我们的策略，分析是否应该买入或者卖出。
        2.在得到买入或卖出信号后，我们在第x+1天挂单进行交易。
        3.前面解决的是信号，那么下面该解决我们到底应该以什么价格买入或者卖出比较好？
        4.目前给句的策略，是用第x天的最高价与最低价的均值，作为买入或者卖出价。
        5.另外，当我们在第x天决定在第x+1天买入后，如果这个股票不争气，在第x+1天的开盘价下跌超过了某个阈值
        6.我们也取消这个买入动作
    '''
    # 当第x天准备买入，发现第x+1天下跌超过2%，就不买了
    g_open_down_rate = 0.02


    class AbuSlippageBuyMeans2(AbuSlippageBuyBase):
        def fit_price(self):
            if (self.kl_pd_buy.open / self.kl_pd_buy.pre_close) < (1 - g_open_down_rate):
                print('跌太多，不买了')
                return np.inf
            self.buy_price = np.mean([self.kl_pd_buy['high'], self.kl_pd_buy['low']])
            return self.buy_price


    # 执行策略
    buy_factors2 = [{'slippage': AbuSlippageBuyMeans2, 'xd': 60, 'class': AbuFactorBuyBreak}, {'xd': 42,
                                                                                               'class': AbuFactorBuyBreak}]
    sell_factors = [sell_factor1, sell_factor2, sell_factor3, sell_factor4]
    # 执行
    capital = AbuCapital(1000000, benchmark)
    orders_pd, action_pd, _ = ABuPickTimeExecute.do_symbols_with_same_factors(['usTSLA'], benchmark, buy_factors2,
                                                                              sell_factors, capital, show=False)

    '''
        6)
        前面，我们针对一只股票：
            1.为了得到什么时候能够买入股票：我们从最基本本的N日趋势跟踪策略，到加入止损止盈策略，再到对止损止盈策略做了修改。
            2.为了得到应该用什么价格买入股票：我们用了滑点买入/卖出的策略
        那么，在下面，我们尝试将我们的策略，运用到多支股票上。
        现在我们暂时采用对多支股票使用相同的因子，进行买入卖出时间点的选择。
    '''
    # 选择多支股票
    choice_symbols = ['usTSLA', 'usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG', 'usWUBA', 'usVIPS']
    capital = AbuCapital(1000000, benchmark)
    # 这里用的是buy_factors，即没有包含滑点买入/卖出策略的buy_factors
    orders_pd, action_pd, _ = ABuPickTimeExecute.do_symbols_with_same_factors(choice_symbols, benchmark, buy_factors,
                                                                              sell_factors, capital, show=False)
    # print(orders_pd[:10])
    # print(action_pd[:10])
    # 可视化结果
    metrics = AbuMetricsBase(orders_pd, action_pd, capital, benchmark)
    metrics.fit_metrics()
    metrics.plot_returns_cmp(only_show_returns=True)

    '''
        7)
        上面metrics的结果：
            买入后卖出的交易数量:88
            买入后尚未卖出的交易数量:5
            胜率:46.5909%
            平均获利期望:6.4254%
            平均亏损期望:-3.9016%
            盈亏比:1.6611
            策略收益: 22.9038%
            基准收益: 52.3994%
            策略年化收益: 11.4747%
            基准年化收益: 26.2518%
            策略买入成交比例:83.8710%
            策略资金利用率比例:31.4213%
            策略共执行503个交易日

        其中，最重要的就是获利期望，亏损期望和胜率，有了这三个值，我们就可以来进行仓位控制。
        其实，我们系统想做这么几件事：
            1. 确定在什么时候买入，什么时候卖出。
            2. 确定买入卖出的时候采用什么价格
            3. 假设手上有资金100w，每次买入股票，花多少钱买入股票；卖出的时候，也考虑卖出多少手上的股票（目前是全卖了）
    '''
    # 设定两个因子
    buy_factors2 = [{'xd': 60, 'class': AbuFactorBuyBreak},
                    {'xd': 42, 'position': {'class': AbuKellyPosition, 'win_rate': metrics.win_rate,
                                            'gains_mean': metrics.gains_mean, 'losses_mean': -metrics.losses_mean},
                     'class': AbuFactorBuyBreak}]
    # 执行
    capital = AbuCapital(1000000, benchmark)
    orders_pd, action_pd, _ = ABuPickTimeExecute.do_symbols_with_same_factors(choice_symbols, benchmark, buy_factors2,
                                                                              sell_factors, capital, show=False)
    print(orders_pd[:10])

    '''
        8)
        1.对于一支股票，从买入卖出点的确认，到买入卖出的价格，以及最后的仓位控制，我们都做了
        2.之前也简单测试了一下多只股票情况下，用同一套策略的可行性。
        3.下面，我们尝试一下多支股票用不的策略。其实很简单，将其封装成dict，然后传入do_symbols_with_diff_factors就好
    '''
    target_symbols=['usSFUN','usNOAH']

    # 对SUFN，向上只采用42日突破
    buy_factor_SFUN=[{'xd':42,'class':AbuFactorBuyBreak}]
    # 对SUFN，向下只采用60日突破
    sell_factor_SFUN=[{'xd':60,'class':AbuFactorSellBreak}]

    # 对NOAH，向上只采用21日突破
    buy_factor_NOAH=[{'xd':21,'class':AbuFactorBuyBreak}]
    # 对NOAH，向下采用42日突破
    sell_factor_NOAH=[{'xd':42,'class':AbuFactorSellBreak}]

    # 构建因子字典
    factor_dict=dict()
    factor_dict['usSFUN']={'buy_factors':buy_factor_SFUN,'sell_factors':sell_factor_SFUN}
    factor_dict['usNOAH']={'buy_factors':buy_factor_NOAH,'sell_factors':sell_factor_NOAH}
    # 执行
    capital = AbuCapital(1000000, benchmark)
    orders_pd, action_pd, _ = ABuPickTimeExecute.do_symbols_with_diff_factors(target_symbols, benchmark, factor_dict,
                                                                              capital)
    # 打印结果
    # print(orders_pd.head())
    # pd.crosstab(orders_pd.buy_factor,orders_pd.symbol)
