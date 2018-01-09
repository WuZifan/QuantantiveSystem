from abc import ABCMeta, abstractclassmethod
import six
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

K_LIVING_DAYS = 27375


def regular_mm(group):
    '''
    正则化，减去最小，除以范围
    :param group:
    :return:
    '''
    return (group - group.min()) / (group.max() - group.min())


class Person(object):
    '''
    表示人类的类
    '''

    def __init__(self):
        self.living = K_LIVING_DAYS
        # 初始化幸福值，财富值，名望值和存活天数
        self.happiness = 0
        self.wealth = 0
        self.fame = 0
        self.living_day = 0

    def living_one_day(self, seek):
        '''
        每天只能有一种追求（seek）
        :param seek:
        :return:
        '''
        # 得到今天独特的收获
        consume_living, happinese, wealth, fame = seek.do_seek_day()
        # 减去生命消耗
        self.living -= consume_living
        # 幸福积累
        self.happiness += happinese
        # 财富积累
        self.wealth += wealth
        # 名望积累
        self.fame += fame
        # 过完这一天
        self.living_day += 1


class BaseSeekDay(six.with_metaclass(ABCMeta, object)):
    '''
    每天不同的追求
    '''

    def __init__(self):
        # 每个追求每天消耗生命的常数
        self.living_consume = 0
        # 每个追求每天幸福指数常数
        self.happiness_base = 0
        # 每个追求每天积累的财富数
        self.wealth_base = 0
        # 每个追求每天积累的名望数
        self.fame_base = 0

        # 每个追求每天消耗生命的可变因素序列
        self.living_factor = [0]
        # 每个追求每天幸福指数的可变因素序列
        self.happiness_factor = [0]
        # 每个追求每天财富指数的可变因素序列
        self.wealth_factor = [0]
        # 每个追求每天名望指数的可变因素序列
        self.fame_factor = [0]

        # 这一生追求了多少天
        self.do_seek_day_cnt = 0
        # 子类进行常数和可变因素序列设置
        self._init_self()

    @abstractclassmethod
    def _init_self(self, *args, **kwargs):
        pass

    @abstractclassmethod
    def _gen_living_days(self, *args, **kwargs):
        pass

    def do_seek_day(self):
        '''
        每一天的具体追求
        :return:
        '''

        # 计算追求这个事要消耗多少生命，生命消耗=living_consume:消耗常数*living_factor：可变序列
        # 当do_seek_day_cnt超过一定值，即追求一定天数后，living_factor也固定
        if self.do_seek_day_cnt >= len(self.living_factor):
            consume_living = self.living_factor[-1] * self.living_consume
        else:
            consume_living = self.living_factor[self.do_seek_day_cnt] * self.living_consume

        # 下面关于幸福，财富，名望都一样

        # 幸福指数=happiness_base:幸福常数 * happiness_factor:可变序列
        if self.do_seek_day_cnt >= len(self.happiness_factor):
            # 超出len(self.happiness_factor), 就取最后一个
            # 由于happiness_factor值由:n—>0 所以happiness_factor[-1]=0
            # 即随着追求一个事物的次数过多后会变的没有幸福感
            happiness = self.happiness_factor[-1] * self.happiness_base
        else:
            # 每个类自定义这个追求的幸福指数常数，以及happiness_factor
            # happiness_factor子类的定义一般是从高－>低变化
            happiness = self.happiness_factor[self.do_seek_day_cnt] * self.happiness_base

        # 财富积累=wealth_base:积累常数 * wealth_factor:可变序列
        if self.do_seek_day_cnt >= len(self.wealth_factor):
            # 超出len(self.wealth_factor), 就取最后一个
            wealth = self.wealth_factor[-1] * self.wealth_base
        else:
            # 每个类自定义这个追求的财富指数常数，以及wealth_factor
            wealth = self.wealth_factor[self.do_seek_day_cnt] * self.wealth_base

        # 权利积累=fame_base:积累常数 * fame_factor:可变序列
        if self.do_seek_day_cnt >= len(self.fame_factor):
            # 超出len(self.fame_factor), 就取最后一个
            fame = self.fame_factor[-1] * self.fame_base
        else:
            # 每个类自定义这个追求的名望权利指数常数，以及fame_factor
            fame = self.fame_factor[self.do_seek_day_cnt] * self.fame_base

        # 追求了多少天
        self.do_seek_day_cnt += 1
        # 返回这个追求一天对生命的消耗，以及得到的幸福，财富和权利
        return consume_living, happiness, wealth, fame


class HealthSeekDay(BaseSeekDay):
    '''
    每一天都追求健康长寿
    '''

    def _init_self(self):
        # 每天都生命的消耗常数，这里是1天
        self.living_consume = 1
        # 每天幸福指数
        self.happiness_base = 1
        # 设定可变因素序列
        self._gen_living_days()

    def _gen_living_days(self):
        # 只生成长为12000的序列，因为下面的happiness_factor序列值由1－>0
        # 所以大于12000次的追求都将只是单纯消耗生命，并不增加幸福指数
        # 即随着做一件事情的次数越来越多，幸福感越来越低，直到完全体会不到幸福
        days = np.arange(1, 12000)
        living_days = np.sqrt(days)
        """
            对生命消耗可变因素序列值由-1->1, 也就是这个追求一开始的时候对生命
            的消耗为负增长，延长了生命，随着追求的次数不断增多对生命的消耗转为正
            数因为即使一个人天天锻炼身体，天天吃营养品，也还是会有自然死亡的那
            一天
        """
        # 这个数列一开始是-1，最后是+1，计算的时候是用来被剪掉
        # 所以一开始是-1，表示能够让你活的长一点（
        self.living_factor = regular_mm(living_days) * 2 - 1
        # 结果在1-0之间 [::-1]: 将0->1转换到1->0
        self.happiness_factor = regular_mm(days)[::-1]


class StockSeekDay(BaseSeekDay):
    """
        StockSeekDay追求财富金钱的一天:
        形象：做股票投资赚钱的事情。
        抽象：追求财富金钱
    """

    def _init_self(self, show=False):
        # 每天对生命消耗的常数＝2，即代表2天
        self.living_consume = 2
        # 每天幸福指数常数＝0.5
        self.happiness_base = 0.5
        # 财富积累常数＝10，默认＝0
        self.wealth_base = 10
        # 设定可变因素序列
        self._gen_living_days()

    def _gen_living_days(self):
        # 只生成10000个序列
        days = np.arange(1, 10000)
        # 针对生命消耗living_factor的基础函数还是sqrt
        living_days = np.sqrt(days)
        # 由于不需要像HealthSeekDay从负数开始，所以直接regular_mm 即:0->1
        # 这里也是掉血越来越快，每词seek wealth后，都会让你减少很多总寿命
        self.living_factor = regular_mm(living_days)

        # 针对幸福感可变序列使用了np.power4，即变化速度比sqrt快
        happiness_days = np.power(days, 4)
        # 幸福指数可变因素会快速递减由1->0
        self.happiness_factor = regular_mm(happiness_days)[::-1]

        """
            这里简单设定wealth_factor=living_factor
            living_factor(0-1), 导致wealth_factor(0-1), 即财富积累越到
            后面越有效率，速度越快，头一个100万最难赚
        """
        self.wealth_factor = self.living_factor


class FameSeekDay(BaseSeekDay):
    """
        FameTask追求名望权力的一天:
        追求名望权力
    """

    def _init_self(self):
        # 每天对生命消耗的常数＝3，即代表3天
        self.living_consume = 3
        # 每天幸福指数常数＝0.6
        self.happiness_base = 0.6
        # 名望权利积累常数＝10，默认＝0
        self.fame_base = 10
        # 设定可变因素序列
        self._gen_living_days()

    def _gen_living_days(self):
        # 只生成12000个序列
        days = np.arange(1, 12000)
        # 针对生命消耗living_factor的基础函数还是sqrt
        living_days = np.sqrt(days)
        # 由于不需要像HealthSeekDay从负数开始，所以直接regular_mm 即:0->1
        self.living_factor = regular_mm(living_days)

        # 针对幸福感可变序列使用了np.power2
        # 即变化速度比StockSeekDay慢但比HealthSeekDay快
        happiness_days = np.power(days, 2)
        # 幸福指数可变因素递减由1->0
        self.happiness_factor = regular_mm(happiness_days)[::-1]

        # 这里简单设定fame_factor=living_factor
        self.fame_factor = self.living_factor


def my_test_life():
    # 初始化我, 你一生的故事：HealthSeekDay
    me = Person()
    # 初始化追求健康长寿
    seek_health = HealthSeekDay()
    # 这里的基本假设是：
    #   1.你每天的追求是会消耗或者提升你能活的最大天数（或者说存货时间池）
    #   2.只要池子里还有天数，你就能继续活（me.living>0)
    #   3.通过me.living_days记录我度过了多少天
    #   4.我每天的追求是会增加池子中的天数，也有可能减少
    #   5.我想知道，当池子为空时，我度过了多少天
    while me.living > 0:
        me.living_one_day(seek_health)

    print('只追求健康长寿快乐活了{}年，幸福指数{},积累财富{},名望权力{}'.format
          (round(me.living_day / 365, 2), round(me.happiness, 2),
           me.wealth, me.fame))
    plt.plot(seek_health.living_factor * seek_health.living_consume)
    plt.plot(seek_health.happiness_factor * seek_health.happiness_base)
    plt.legend(['living_factor', 'happiness_factor'], loc='best')
    plt.show()

    ####只追求金钱
    # 初始化我, 你一生的故事：StockSeekDay
    me = Person()
    # 初始化追求财富金钱
    seek_stock = StockSeekDay()
    while me.living > 0:
        # 只要还活着，就追求财富金钱
        me.living_one_day(seek_stock)

    print('只追求财富金钱活了{}年，幸福指数{}, 积累财富{}, 名望权力{}'.format
          (round(me.living_day / 365, 2), round(me.happiness, 2),
           round(me.wealth, 2), me.fame))

    plt.plot(seek_stock.living_factor * seek_stock.living_consume)
    plt.plot(seek_stock.happiness_factor * seek_stock.happiness_base)
    plt.legend(['living_factor', 'happiness_factor'], loc='best')
    plt.show()

    ####只只求名望
    # 初始化我, 你一生的故事：FameSeekDay
    me = Person()
    # 初始化追求名望权力
    seek_fame = FameSeekDay()
    while me.living > 0:
        # 只要还活着，就追求名望权力
        me.living_one_day(seek_fame)

    print('只追求名望权力活了{}年，幸福指数{}, 积累财富{}, 名望权力{}'.format
          (round(me.living_day / 365, 2), round(me.happiness, 2),
           round(me.wealth, 2), round(me.fame, 2)))

    plt.plot(seek_fame.living_factor * seek_fame.living_consume)
    plt.plot(seek_fame.happiness_factor * seek_fame.happiness_base)
    plt.legend(['living_factor', 'happiness_factor'], loc='best')
    plt.show()


def my_life(weights):
    '''
    想要知道如何分配时间才能让幸福最大
    :param weights:
    :return:
    '''
    # 追求健康长寿快乐
    seek_health = HealthSeekDay()
    # 追求财富金钱
    seek_stock = StockSeekDay()
    # 追求名望权力
    seek_fame = FameSeekDay()

    # 放在一个list中对对应下面np.random.choice中的index[0, 1, 2]
    seek_list = [seek_health, seek_stock, seek_fame]

    # 初始化我
    me = Person()
    # 加权随机抽取序列。80000天肯定够了, 80000天快220年了。。。
    seek_choice = np.random.choice([0, 1, 2], 80000, p=weights)

    while me.living > 0:
        # 追求从加权随机抽取序列已经初始化好的
        seek_ind = seek_choice[me.living_day]
        seek = seek_list[seek_ind]
        # 只要还活着，就追求
        me.living_one_day(seek)
    return round(me.living_day / 365, 2), round(me.happiness, 2), \
           round(me.wealth, 2), round(me.fame, 2)


def montekalo_method():
    '''
    用蒙特卡洛方法求解
    :return:
    '''
    # 用来保存结果
    result = []
    # 做2000次实验
    for _ in range(0, 2000):
        weights = np.random.random(3)
        weights /= np.sum(weights)
        result.append((weights, my_life(weights)))

    # result中tuple[1]=my_life返回的结果, my_life[1]=幸福指数，so->x[1][1]
    sorted_scores = sorted(result, key=lambda x: x[1][1], reverse=True)

    # 将最优weights带入my_life，计算得到最优结果
    living_day, happiness, wealth, fame = my_life(sorted_scores[0][0])

    print('活了{}年，幸福指数{}, 积累财富{}, 名望权力{}'.format
          (living_day, happiness, wealth, fame))

    print('人生最优权重：追求健康{:.3f},追求财富{:.3f},追求名望{:.3f}'.format(
        sorted_scores[0][0][0], sorted_scores[0][0][1],
        sorted_scores[0][0][2]))


def minimize_happiness_global(weigths):
    if np.sum(weigths) != 1:
        return 0
    return -my_life(weigths)[1]


def global_search():
    # 下面函数表示，我做20次独立实验，从[0,1,2]这三个数字中抽取一个出来
    # 每次实验中，0，1，2三个数被抽取的概率分别为0.2，0.3，0.5
    opt_global = sco.brute(minimize_happiness_global, ((0, 1.1, 0.1), (0, 1.1, 0.1), (0, 1.1, 0.1)))

    # 将最优weights带入my_life，计算得到最优结果
    living_day, happiness, wealth, fame = my_life(opt_global)

    print('活了{}年，幸福指数{}, 积累财富{}, 名望权力{}'.format
          (living_day, happiness, wealth, fame))

    print('人生最优权重：追求健康{:.3f},追求财富{:.3f},追求名望{:.3f}'.format(
        opt_global[0], opt_global[1], opt_global[2]))

def minimizi_happiness_local(weights):
    print(weights)
    return -my_life(weights)[1]

if __name__ == '__main__':
    # my_test_life()
    method = 'SLSQP'
    # 约束
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # 构造需要被调节的参数
    bounds = tuple((0, 1) for i in range(3))
    # 初始化猜测参数
    guess=[0.5,0.2,0.3]

    opt_local=sco.minimize(minimizi_happiness_local,guess,method=method,bounds=bounds,constraints=constraints)

    print(opt_local)
