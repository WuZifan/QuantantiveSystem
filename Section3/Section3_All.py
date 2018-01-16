import timeit
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs


if __name__ == '__main__':

    # 1、关于作用点
    normal_list=[1,1,1,1,1]
    np_list=np.ones(5)
    # 普通数组的乘法是作用在数组这个整体上
    print(normal_list*3)
    # np数组的乘法是作用在数组的每个元素上
    print(np_list*3)

    # 2、关于初始化操作
    # 初始化的集中操作 zeros，ones，empyt，eye，ones_like,zeros_like
    # 另外，可以从normal_list得到np的array
    normal=[[1,2,3],[4,5,6]]
    np_array=np.array(normal)
    print(type(np_array))
    print(np_array)

    # 等间距的生成np。array
    # 计算公式是linspace(a,b,c),开头是a，结尾是b，中间每个数为a+n*(b-a)/(c-1),n属于1~c-1
    equal_step=np.linspace(0,9,5)
    print(equal_step)

    # 3、关于正态分布
    ###我们生成一下正态分布数据
    stock_cnt=200
    view_days=504
    # 关于正态分布
    # 这里注意下生成的正态分布的股票数据
    # 是指，股票每天的票价都是从符合正太分布的数列中随机选取
    # 其概率密度函数符合单峰的状态
    # 而我们把股票的价格画曲线，是看不到单峰图的
    stock_day_change=np.random.standard_normal((stock_cnt,view_days))

    # 4、关于切片
    # 对于[0:2,0:5],表示获得第一条和第二条股票的前5天数据，返回的可以看做是一个list套list的结构
    # 它的类型是一个np.array,array中每个元素还是一个np。array
    print(stock_day_change[0:2,0:5])
    # 对于[0:1,:][0],这个操作将最外层的list去掉，表示只取第一条股票
    # 它是一个np.array类型的数据，里面每个元素是一个integer
    stock_one=stock_day_change[0:2,:][0]
    print(stock_one[:5])

    # 简单画图
    # plt.plot(stock_one)
    # plt.show()


    # 5、关于引用
    # 发现当我改变了stock_one中数据后，stock_day_change中的数据也变化
    # 因为在np.array的切片返回值中，每个元素都是一个引用
    # 所以改变切片的值也会改变原来array的值
    # 只有试用copy操作，才不会改变
    stock_one=stock_one.copy()
    stock_one[0]=1
    print(stock_day_change[0:1,0:2])
    print(stock_one[0:2])


    # 6、关于数据转化和规整
    # 其实就是使用了对于numpy中的array而言，操作都会作用在每个元素上，这一个特性
    stock_temp=stock_day_change[0:2,0:5].copy()
    # 注意这里的astype没有对stock_temp进行操作，而是构建了新的array
    # 所以不会影响原array
    print(stock_temp.astype(int))
    # 这个np.round也不会影响原数组
    # 注意这里用的是np.round，而不是python自带的round
    print(np.round(stock_temp,2)[0][0])
    print(stock_temp)

    # 7、关于mask
    #  先通过对array中每一个元素进行比较操作，返回一个和原array大小相同，但全部由Ture，False组成的array
    #  在利用这个新的array去和原数组进行操作，得到那些True位置上对应的数
    temp_mask=stock_temp>0.5
    print(temp_mask)
    print(stock_temp[temp_mask])
    # 在一行内完成：1.取到大于1或者小于-1的数，并将它们赋值为0
    print(stock_temp)
    # 注意这里逻辑符号只能用 |，&，不能用and或者or
    stock_temp[(stock_temp>1) | (stock_temp<-1)]=0
    print(stock_temp)

    # 8、关于通用序列函数
    stock_temp=stock_day_change[0:2,0:5].copy()
    # 8.1 判断序列中元素是否都是true
    print(np.all(stock_temp>0))
    # 8.2 判断序列中元素是否存在大于0的
    print(np.any(stock_temp>0))
    # 8.3 对于两个序列中的元素进行两两比较，留下较大(或者较小的）序列
    stock_last2=stock_day_change[-2:,0:5]
    print(stock_temp)
    print(stock_last2)
    print(np.maximum(stock_temp,stock_last2))
    print(np.minimum(stock_temp,stock_last2))

    # 8.4 重复的值只留下一个，和set功能类似
    print(np.unique(np.array([1,1,0,0,0,2])))
    # 8.5 计算差值，axis=0表示比较行之间不同，axis=1表示比较列之间不同
    # 默认比较列,即第0列和第一列，第一列和第二列...
    print(stock_temp)
    print(np.diff(stock_temp))
    print(np.diff(stock_temp,axis=0))

    # 8.6 np的三目表达式，如果条件成立，执行状况1，否则执行状况2
    # 条件成立就赋值为1，不然赋值为0
    print(np.where(stock_last2>0.5,1,0))
    # 条件成立就赋值为1，不就保持怨言
    print(np.where(stock_last2>0.5,1,stock_last2))
    # 复合条件判断，比如说and
    print(np.where(np.logical_and(stock_last2>0.5,stock_last2<1),1,0))

    # 9、关于统计函数
    stock_day_four = stock_day_change[0:4, 0:4]
    # 9.1 前4只股票分别的最大涨幅
    print(stock_day_four)
    # axis什么都不写，表示统计全局最大值
    # 这里的axis需要注意，axis=0，表示返回结果以行向量的形式返回，那么计算的就是每一列的最大值
    print(np.max(stock_day_four,axis=0))
    # axis=1，表示计算结果以列向量形式返回，计算的是每一行的最大值
    print(np.max(stock_day_four,axis=1))
    # 类似的操作还有 均值mean，标准差std，最小值min等等
    # 当我们不想获得值，而是想获得哪一天得到极值，可以用argmax，argmin
    print(np.argmax(stock_day_four,axis=1))

    # 10、关于正态分布
    stock_day_one=stock_day_change[0]
    # 表示画直方图，一共画50个方块，对应于图中蓝色部分
    # 每一个方块表示在这个范围内的点有多少
    plt.hist(stock_day_one,bins=50,normed=True)
    # 默认分成50个点
    fit_linspace=np.linspace(stock_day_one.min(),stock_day_one.max(),num=50)
    # 下面是用正态分布取去拟合这个直方图
    day_one_mean=np.mean(stock_day_one)
    day_one_std=np.std(stock_day_one)
    # 这里注意，pdf就是图中黄色曲线上的点的值，不代表任何意思，不代表概率值！
    # 正态分布是一个概率密度函数，只有曲线下的面积，才代表概率
    pdf=scs.norm(day_one_mean,day_one_std).pdf(fit_linspace)
    plt.plot(fit_linspace,pdf)
    # plt.show()

    # 简单验证了一下，用横坐标之间的差表示宽
    # 纵坐标两点间的均值表示高，简单计算了一下曲线下的面积，差不多是1，就是概率
    # 所以曲线上的点，其实真心没什么意思
    width=np.diff(fit_linspace,axis=0)
    height=(pdf[1:]+pdf[:-1])/2
    print(sum(width*height))

    # 11、简单例子，正态分布买入策略
    # 最后50天作为验证数据
    keep_days=50
    # 取出每只股票前454天的数据（因为一共生成504天）
    stock_day_change_test=stock_day_change[:,:-keep_days]
    print(np.shape(stock_day_change_test))
    # 计算所有股票的总涨跌幅，选出跌的最多的三只股票的序号，所以用的argsorted（就是总和最小的三只）
    stock_array=np.argsort(np.sum(stock_day_change_test,axis=1))[:3]

    def show_buy_lower(stock_ind):
        '''
        :param stock_ind:股票序号
        :return:
        '''
        # 设置一行两列的图表
        _,axs=plt.subplots(nrows=1,ncols=2)
        # 绘制前454天走势图
        axs[0].plot(np.arange(0,view_days-keep_days),stock_day_change_test[stock_ind].cumsum())

        # 这个cumsum是求累积和
        # 比如cumsum([1,3,5),返回值是[1，4，9]
        cs_buy=stock_day_change[stock_ind][view_days-keep_days:].cumsum()
        # 绘制从454天到504天的走势图
        axs[1].plot(np.arange(view_days-keep_days,view_days),cs_buy)
        plt.show()
        return cs_buy[-1]

    profit=0
    for stock_ind in stock_array:
        profit+=show_buy_lower(stock_ind)
    print('总盈亏为%s'%profit)
    # 简答总结一些上面的策略为什么会盈利：
    #  1. 我们生成的涨跌幅数据是标准的正态分布。均值为0，方差为1
    #  2. 表示，每次涨跌幅小于0，即股票下跌的概率为50%，上涨也为50%
    #  3. 那么，对于由n个独立同分布（符合标准正态分布）的随机变量，生成的长为n的数列
    #  4. 如果前k个值得和小于0，那么后n-k个值得和大于0的概率会比较大。（这是大数定理把，因为均值要是0）


    # 12、基于伯努利的交易
    gamblers=100

    def casino(win_rate,win_once=1,loss_once=1,commission=0.01):
        '''

        :param win_rate: 获胜概率
        :param win_once: 每次赢的收益
        :param loss_once: 每次输的损失
        :param commission: 手续费
        :return:
        '''
        my_money=10000
        play_cnt=100000
        commission=commission
        for _ in np.arange(0,play_cnt):
            w=np.random.binomial(1,win_rate)
            if w:
                my_money+=win_once
            else:
                my_money-=loss_once

            my_money-=commission

            if my_money<0:
                break

        return my_money

    # 加速跑一下
    import numba as nb
    casino=nb.jit(casino)

    heaaven_moneys=[casino(0.5,commission=0) for _ in np.arange(0,gamblers)]
    cheat_moneys=[casino(0.4,commission=0) for _ in np.arange(0,gamblers)]
    commission_moneys=[casino(0.5,commission=0.01) for _ in np.arange(0,gamblers)]

    plt.hist(heaaven_moneys,bins=30)
    plt.show()