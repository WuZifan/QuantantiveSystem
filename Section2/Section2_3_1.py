from collections import namedtuple
from collections import OrderedDict


class StockTradeDays(object):
    # 构造函数
    def __init__(self, price_array, start_date, date_array=None):
        # 初始化私有的价格序列，日期序列和涨幅序列
        self.__price_array = price_array
        self.__date_array = self.__init_days(start_date, date_array)
        self.__change_array = self.__init_change()
        # 初始化股票序列
        self.stock_dict = self._init_stock_dict()

    def __init_change(self):
        '''
            protected方法：
            从price_array中生成涨幅

        :return: change_array，涨幅
        '''
        price_float_array=[float(price_str) for price_str in self.__price_array]
        # 形成两个错开的价格队列
        pp_array=[(price1,price2) for price1,price2 in zip(price_float_array[:-1],price_float_array[1:])]
        # 计算涨幅
        change_array=[round((b-a)/a,3) for a,b in pp_array]
        # 将第一天涨幅设置为0
        change_array.insert(0,0)
        return change_array

    def __init_days(self,start_date,date_array):
        '''
            protected方法
        :param start_date: 给定初始日期
        :param date_array: 日期序列
        :return:
        '''
        if date_array is None:
            # 如果初始日期为空，那么就用start_date和price_array来确定初始日期
            date_array=[str(start_date+ind) for ind,_ in enumerate(self.__price_array)]
        else:
            # 如果外部给了时间序列，那就转换成str存好
            date_array=[str(date) for date in date_array]
        return date_array

    def _init_stock_dict(self):
        '''
        试用namedtuple，orderdict将结果合并
        :return:
        '''
        stock_namedtuple=namedtuple('stock',('date','price','change'))
        stock_dict=OrderedDict((date,stock_namedtuple(date,price,change)) for date,price,change in
                               zip(self.__date_array,self.__price_array,self.__change_array))
        return stock_dict

    def filter_stock(self,want_up=True,want_calc_sum=False):
        '''
            筛选结果集
        :param want_up: 是否筛选上涨
        :param want_calc_sum: 是否计算涨跌和
        :return:
        '''
        # <为真时的结果> if <判定条件> else <为假时的结果>
        filter_func=(lambda p_day:p_day.change>0) if want_up else(lambda p_day:p_day.change<0)

        # 来对其进行筛选
        # 因为是namedtuple，所以可以用.change取到chagne的值
        want_days=list(filter(filter_func,self.stock_dict.values()))

        # 如果不计算上涨和，那么就把哪几天上涨的范湖
        if not want_calc_sum:
            return want_days

        change_sum=0.0
        for day in want_days:
            change_sum+=day.change
        return change_sum

    def __str__(self):
        return str(self.stock_dict)

    __repr__=__str__

    def __iter__(self):
        '''
            使用生成器来进行迭代
        :return:
        '''
        for key in self.stock_dict:
            yield self.stock_dict[key]

    def __getitem__(self,ind):
        '''
        拿到某一天的交易信息
        :param ind:
        :return:
        '''
        date_key=self.__date_array[ind]
        return self.stock_dict[date_key]

    def __len__(self):
        return len(self.stock_dict)

if __name__ == '__main__':
    price_array = '30.14,29.58,26.36,32.56,32.82'.split(',')
    date_base = 20170118
    # 从StockTradeDays类初始化一个实例对象trade_days，内部会调用__init__
    trade_days = StockTradeDays(price_array, date_base)
    # 打印对象信息
    # print(trade_days)
    for day in trade_days:
        print(day)

    print(trade_days.filter_stock())

    # 因为可迭代，一定程度上可以当做是list来用
    print(trade_days[-1])