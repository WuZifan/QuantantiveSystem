import six
from abc import ABCMeta, abstractclassmethod
from Section2.Section2_3_1 import StockTradeDays
from abupy import ABuSymbolPd
import itertools


class TradeStrategyBase(six.with_metaclass(ABCMeta, object)):
    '''
        交易策略的抽象类
    '''

    # *arg表示可以有任意多个非指定参数，比如day，price，等等
    # **kwargs表示可以有任意多个制定参数，比如isUp=False，isDown=True等等

    def but_strategy(self, *arg, **kwargs):
        # 买入的基类
        pass

    def sell_strategy(self, *arg, **kwargs):
        # 卖出的基类
        pass


class TradeStrategy1(TradeStrategyBase):
    '''
        交易策略1：追涨，当股价上涨超过一个阈值（7%）时，
        买入并持有s_keep_stock_threshold天
    '''
    s_keep_stock_threshold = 20

    def __init__(self):
        '''
        初始化一些参数
        '''
        self.keep_stock_day = 0
        self.__buy_change_threshold = 0.07

    def buy_strategy(self, trade_ind, trade_day, trade_days):
        if self.keep_stock_day == 0 and trade_day.change > self.__buy_change_threshold:
            # 当还没有持有股票
            # 并且股票上涨超过阈值时，买入股票
            self.keep_stock_day += 1
        elif self.keep_stock_day > 0:
            # 已经持有过股票，那么持有天数加1
            self.keep_stock_day += 1

    def sell_strategy(self, trade_ind, trade_day, trade_days):
        # 当持有股票天数超过规定天数时，卖出
        if self.keep_stock_day >= TradeStrategy1.s_keep_stock_threshold:
            self.keep_stock_day = 0

    # 相当于 buy_change_threshold的get方法
    @property
    def buy_change_threshold(self):
        return self.__buy_change_threshold

    # 相当于 buy_change_threshold的set方法
    @buy_change_threshold.setter
    def buy_change_threshold(self, buy_change_threshold):
        if not isinstance(buy_change_threshold, float):
            raise TypeError('buy_change_threshold must be float')
        self.__buy_change_threshold = round(buy_change_threshold, 2)


class TradeStrategy2(TradeStrategyBase):
    '''
        交易策略2：均值回复策略，当股票连续连个交易日下跌
        且下跌幅度超过默认的s_buy_change_threshold(-10%)时
        买入股票并持有s_keep_stock_threshold（10）天

    '''
    s_keep_stock_threshold = 10
    s_buy_change_threshold = -0.10

    def __init__(self):
        self.keep_stock_day = 0

    def buy_strategy(self, trade_ind, trade_day, trade_days):
        '''
        :param trade_ind: 交易日下标
        :param trade_day: 当天交易信息
        :param trade_days: 交易信息序列
        :return:
        '''
        if self.keep_stock_day == 0 and trade_ind >= 1:
            '''
                当手上还没有股票，且从第二天开始；
                因为第一天没有前一天数据，所以只能从第二天开始
            '''
            # 如果当天下跌
            today_down = trade_day.change < 0
            # 如果昨天下跌
            yesterday_down = trade_days[trade_ind - 1].change < 0
            # 两天的总跌幅
            down_rate = trade_day.change + trade_days[trade_ind - 1].change
            # 因为是负数，所以注意负号
            if today_down and yesterday_down and down_rate <= TradeStrategy2.s_buy_change_threshold:
                self.keep_stock_day += 1
        elif self.keep_stock_day > 0:
            self.keep_stock_day += 1

    def sell_strategy(self, trade_ind, trade_day, trade_days):
        if self.keep_stock_day > TradeStrategy2.s_keep_stock_threshold:
            self.keep_stock_day = 0

    @classmethod
    def set_keep_stock_threshold(cls, keep_stock_threshold):
        cls.s_keep_stock_threshold = keep_stock_threshold

    @staticmethod
    def set_buy_change_threshold(buy_change_threshold):
        TradeStrategy2.s_buy_change_threshold = buy_change_threshold


class TradeLoopBack(object):
    '''
        回测系统
    '''

    def __init__(self, trade_days, trade_strategy):
        '''
        :param trade_days: 交易数据序列
        :param trade_strategy: 交易策略
        '''
        self.trade_days = trade_days
        self.trade_strategy = trade_strategy
        self.profit_array = []

    def execute_trade(self):
        '''
        执行回测
        回测的意义：
        https://www.investopedia.com/terms/b/backtesting.asp
        :return:
        '''
        for ind, day in enumerate(self.trade_days):
            '''
                时间驱动
            '''
            if self.trade_strategy.keep_stock_day > 0:
                # 如果当天持有股票，那么将当天的股票收入加入盈亏队列
                self.profit_array.append(day.change)

            # hasattr用来判断对象self.trade_strategy 是否实现了方法'buy_strategy'
            if hasattr(self.trade_strategy, 'buy_strategy'):
                # 执行买入策略，看看今天是否需要买入
                self.trade_strategy.buy_strategy(ind, day, self.trade_days)

            if hasattr(self.trade_strategy, 'sell_strategy'):
                # 执行卖出策略，看今天要不要卖
                self.trade_strategy.sell_strategy(ind, day, self.trade_days)


def do_strateg1(trade_days):
    '''
    执行策略1
    :param trade_days:
    :return:
    '''
    # 策略1
    trade_strategy1 = TradeStrategy1()

    # 执行回测
    trade_loop_back = TradeLoopBack(trade_days, trade_strategy1)
    trade_loop_back.execute_trade()

    # 查看回测结果
    print(trade_loop_back.profit_array)
    print(sum(trade_loop_back.profit_array))

    # 绘图查看
    plot_array(trade_loop_back.profit_array)

def do_strategy1_changeParameter(trade_days):
    '''
    执行策略1，并修改参数
    :param trade_days:
    :return:
    '''
    # 修改策略
    trade_strategy1 = TradeStrategy1()
    # 上涨10%才买（10%不就涨停嘛。。基本很难看见，所以没法获利很正常
    # 没有一天是大于10%的。。。。换8%试试吧
    # 这里就利用property，来对值进行修改
    trade_strategy1.buy_change_threshold = 0.08

    # 执行回测
    trade_loop_back2 = TradeLoopBack(trade_days, trade_strategy1)
    trade_loop_back2.execute_trade()

    # 查看回测结果
    print(trade_loop_back2.profit_array)
    print(sum(trade_loop_back2.profit_array))

    # 绘图查看
    plot_array(trade_loop_back2.profit_array)

def do_strategy2(trade_days):
    '''
    执行策略2
    :param trade_days:
    :return:
    '''
    # 第二套策略
    trade_strategy2 = TradeStrategy2()

    # 执行回测
    trade_loop_back3 = TradeLoopBack(trade_days, trade_strategy2)
    trade_loop_back3.execute_trade()

    # 查看回测结果
    print(trade_loop_back3.profit_array)
    print(sum(trade_loop_back3.profit_array))

    # 绘图查看
    plot_array(trade_loop_back3.profit_array)

def do_strategy_changeParameter(trade_days):
    '''
    执行策略2，并修改参数
    :param trade_days:
    :return:
    '''
    # 修改第二套策略参数
    trade_strategy3 = TradeStrategy2()
    TradeStrategy2.set_keep_stock_threshold(20)
    TradeStrategy2.set_buy_change_threshold(-0.08)
    # 执行回测
    trade_loop_back4 = TradeLoopBack(trade_days, trade_strategy3)
    trade_loop_back4.execute_trade()

    # 查看回测结果
    print(trade_loop_back4.profit_array)
    print(sum(trade_loop_back4.profit_array))

    # 绘图查看
    plot_array(trade_loop_back4.profit_array)

# 画图方法
def plot_array(array):
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    sns.set_context(rc={'figure.figsize': (14, 7)})
    plt.plot(np.array(array).cumsum())
    plt.show()


# 寻找最优参数
def calc(keep_stock_threshold, buy_change_threshold, trade_days):
    '''

    :param keep_stock_threshold: 涨跌幅的阈值
    :param buy_change_threshold: 持股天数阈值
    :param trade_days:交易序列
    :return: 盈亏情况，输入的持股天数，输入的涨跌幅阈值
    '''
    # 初始化策略，并修改参数
    trade_strategy2 = TradeStrategy2()
    TradeStrategy2.set_keep_stock_threshold(keep_stock_threshold)
    TradeStrategy2.set_buy_change_threshold(buy_change_threshold)

    # 执行回测
    trade_loop_back=TradeLoopBack(trade_days,trade_strategy2)
    trade_loop_back.execute_trade()

    # 计算最终盈亏值
    profit=sum(trade_loop_back.profit_array)

    return profit,keep_stock_threshold,buy_change_threshold



if __name__ == '__main__':
    # 用特斯拉的数据来做演示
    # 获取特斯拉的数据
    price_array = ABuSymbolPd.make_kl_df('TSLA', n_folds=2).close.tolist()
    date_array = ABuSymbolPd.make_kl_df('TSLA', n_folds=2).date.tolist()
    date_base = 20170118

    # 构建股票交易序列
    trade_days = StockTradeDays(price_array, date_base, date_array)

    print(calc(20,-0.08,trade_days))

    ########寻找最优参数#########
    # 类似于grid-search
    keep_stock_list=range(2,30,2)
    buy_change_list=[buy_change/100.00 for buy_change in range(-5,-16,-1)]

    result=[]
    for keep_stock_threshold,buy_change_threshold in itertools.product(keep_stock_list,buy_change_list):
        result.append(calc(keep_stock_threshold,buy_change_threshold,trade_days))

    print(sorted(result)[::-1][:10])
