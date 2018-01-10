import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    trade_day = 100


    # 1、涨跌规则1
    def gen_stock_price_array():
        '''
        这里的涨跌规则是：
            1.前一天上涨，那么今天也涨；前一天下跌，那么今天也跌
            2.上涨和下跌都只有5%
        :return:
        '''
        # 初始化每天都是1元
        price_array = np.ones(trade_day)
        # 生成100个交易日走势

        for ind in np.arange(0, trade_day - 1):
            # 以下代码是通过今天决定明天的股价
            if ind == 0:
                # binomial三个参数为n,p,size,表示成功次数，成功概率和总实验数
                win = np.random.binomial(1, 0.5)
            else:
                win = price_array[ind] > price_array[ind - 1]

            if win:
                price_array[ind + 1] = (1 + 0.05) * price_array[ind]

            else:
                price_array[ind + 1] = (1 - 0.05) * price_array[ind]
        return price_array


    fig, axis = plt.subplots(ncols=2, nrows=1)
    price_array1 = gen_stock_price_array()
    price_array1_ex = gen_stock_price_array()
    axis[0].plot(np.arange(0, trade_day), price_array1)
    axis[1].plot(np.arange(0, trade_day), price_array1_ex)
    plt.show()

    # 2、涨跌规则第二部分
    trade_day = 252


    def gen_stock_price_array2():
        '''
        第二阶段有252天，用上一阶段的最后一天的值初始化这个252天
        并把这252天接在上一阶段之后
        :return:
        '''
        price_array = np.concatenate((price_array1, np.ones(trade_day) * price_array1[-1]), axis=0)

        # 开始处理这一阶段的股价：
        for ind in np.arange(len(price_array1) - 1, len(price_array) - 1):
            # 获取当前交易日（在内）的前4天数据
            last4 = price_array[ind - 3:ind + 1]
            # 如果还没有到尽头,且连涨三天
            if len(last4) == 4 and last4[-1] > last4[-2] and last4[-2] > last4[-3] and last4[-3] > last4[-4]:
                # 那么下跌概率为0.55，即继续涨的概率为0.45
                win = np.random.binomial(1, 0.45)
            elif len(last4) == 4 and last4[-1] < last4[-2] and last4[-2] < last4[-3] and last4[-3] < last4[-4]:
                # 如果连跌三天
                win = np.random.binomial(1, 0.80)
            else:
                # 当天涨跌仍然只和前一天相关
                win = price_array[ind] > price_array[ind - 1]

            if win:
                price_array[ind + 1] = (1 + 0.05) * price_array[ind]

            else:
                price_array[ind + 1] = (1 - 0.05) * price_array[ind]

        return price_array


    import itertools

    fig, axis = plt.subplots(ncols=3, nrows=3)
    ax_list = list(itertools.chain.from_iterable(axis))
    # 趋势太小看不清楚
    for ind in range(0, len(ax_list)):
        ax_list[ind].plot(gen_stock_price_array2())
    plt.show()
    # 看单个的就很清楚了
    price_array2 = gen_stock_price_array2()
    plt.plot(price_array2)
    plt.show()

    # 3、股票涨跌的第三种情况
    trade_day = 252 * 3


    def gen_stock_price_array3():
        '''
               第三阶段有252天，用上一阶段的最后一天的值初始化这个252天
               并把这252天接在上一阶段之后
               :return:
               '''
        price_array = np.concatenate((price_array2, np.ones(trade_day) * price_array2[-1]), axis=0)

        # 开始处理这一阶段的股价：
        for ind in np.arange(len(price_array1) - 1, len(price_array) - 1):
            # 获取当前交易日（在内）的前4天数据
            last4 = price_array[ind - 3:ind + 1]
            # 如果还没有到尽头,且连涨三天
            if len(last4) == 4 and last4[-1] > last4[-2] and last4[-2] > last4[-3] and last4[-3] > last4[-4]:
                # 那么下跌概率为0.55，即继续涨的概率为0.45
                win = np.random.binomial(1, 0.45)
            elif len(last4) == 4 and last4[-1] < last4[-2] and last4[-2] < last4[-3] and last4[-3] < last4[-4]:
                # 如果连跌三天
                # 那么上涨的概率是80%
                win = np.random.binomial(1, 0.80)
                # 如果这样都没涨
                if not win:
                    # 股灾，股价直接跌50%
                    price_array[ind + 1] = (1 - 0.5) * price_array[ind]
                    # 如果股价太低
                    if price_array[ind + 1] <= 0.1:
                        # 股价太低，直接退市
                        price_array[ind + 1:] = 0
                        break
                    else:
                        # 进入下一个循环，不用进行后面的操作了
                        continue
            else:
                # 当天涨跌仍然只和前一天相关
                win = price_array[ind] > price_array[ind - 1]

            if win:
                price_array[ind + 1] = (1 + 0.05) * price_array[ind]

            else:
                price_array[ind + 1] = (1 - 0.05) * price_array[ind]

            if price_array[ind + 1] <= 0.1:
                # 股价太低，直接退市
                price_array[ind + 1:] = 0
                break

        return price_array


    price_array3 = gen_stock_price_array3()
    plt.plot(gen_stock_price_array3())
    plt.show()

    '''
     上面模拟了一只股票3年的股价状况
     下面我们用不同的交易策略在这上面进行测试，看看哪个交易策略能够盈利。
    '''


    def execute_trade(cash, buy_rate):
        '''

        :param cash: 最初持有现金数目
        :param buy_rate: 每次买入所花的钱，即持仓比例
        :return:
        '''
        commission = 5  # 手续费
        stock_cnt = 0  # 持有股票数
        keep_day = 0  # 持股天数

        # 资产序列
        capital = []
        # 从price_array3的第352天后开始
        for ind in np.arange(252, len(price_array3) - 1):
            if stock_cnt > 0:
                # 如果持有股票，那么增加持股天数
                keep_day += 1
            if stock_cnt > 0 and keep_day == 3:
                # 如果持有股票，且持有了3天
                # 那么卖出股票
                cash += price_array3[ind] * stock_cnt
                # 扣除手续费
                cash -= commission
                # 如果资产为负的
                if cash <= 0:
                    capital.append(0)
                    print('爆仓了')
                    break

            # 获取包括今天在内的5个交易日的数据
            last5 = price_array3[ind - 4:ind + 1]
            # 买入策略：
            # 1.当没有持有股票，且第一个交易日上涨，后三个交易日下跌，买入
            if stock_cnt == 0 and len(last5) == 5 \
                    and last5[1] > last5[0] \
                    and last5[2] < last5[1] \
                    and last5[3] < last5[2] \
                    and last5[4] < last5[3]:
                cash -= commission
                buy_cash = cash * buy_rate
                cash -= buy_cash
                # 计算持了多少股
                stock_cnt += buy_cash / price_array3[ind]

                if stock_cnt < 1:
                    # 表示1股都买不起
                    capital.append(0)
                    print('爆仓')
                    break
                keep_day = 0

            # 计算当前总资产
            capital.append(cash + (stock_cnt * price_array3[ind]))
        return capital


    pig_one_cash = 10000
    buy_rate = 1
    pig_one_captical = execute_trade(pig_one_cash, buy_rate)
    print('猪1的总资产:{}'.format(pig_one_captical[-1]))
    print('猪1的最高值：{}'.format(max(pig_one_captical)))
    plt.plot(pig_one_captical)
    plt.show()

    pig_two_cash = 10000
    buy_rate = 0.6
    pig_two_captical = execute_trade(pig_two_cash, buy_rate)
    print('猪2的总资产:{}'.format(pig_two_captical[-1]))
    print('猪2的最高值：{}'.format(max(pig_two_captical)))
    plt.plot(pig_two_captical)
    plt.show()

    pig_trhee_cash = 10000
    buy_rate = 0.13
    pig_three_captical = execute_trade(pig_trhee_cash, buy_rate)
    print('猪3的总资产:{}'.format(pig_three_captical[-1]))
    print('猪3的最高值：{}'.format(max(pig_three_captical)))
    plt.plot(pig_three_captical)
    plt.show()
