import matplotlib.pyplot as plt
from abupy import ABuSymbolPd
import matplotlib.finance as mpf




def plot_demo(tsla_df,axs=None,just_serise=False):
    '''
    1.1 利用matplot绘制series，numpy以及list数据
    :param axs:子画布
    :param just_serise:是否只画series类型的线
    :return:
    '''
    # 创建画布
    drawer=plt if axs is None else axs
    # 绘制serise对象
    drawer.plot(tsla_df.close)

    if not just_serise:
        # 画numpy对象
        # Dataframe.index名称.values，返回的就是numpy对象
        drawer.plot(tsla_df.close.index,tsla_df.close.values+10,c='g')

        # 画list对象
        drawer.plot(tsla_df.close.index.tolist(),(tsla_df.close.values+20).tolist(),c='b')

    plt.xlabel('time')
    plt.ylabel('close')
    plt.title('TSLA')
    plt.grid(True)



if __name__ == '__main__':
    tsla_df=ABuSymbolPd.make_kl_df('TSLA',n_folds=2)
    print(tsla_df.head())

    # 1.1 利用matplot绘制三种不同类型的数据
    # plot_demo(tsla_df)
    # plt.show()

    # 1.2 子画布以及loc
    # 生成2*2的画布，上面可以有4个图
    # _,axs=plt.subplots(nrows=2,ncols=2)
    # for i in range(0,2):
    #     for j in range(0,2):
    #         drawer=axs[i][j]
    #         plot_demo(tsla_df,axs=drawer)
    #         drawer.legend(['Series','Numpy','List'],loc='best')
    #
    # plt.show()

    # 1.3 K线图绘制
    __colorup__='red'
    __colordown__='green'
    tsla_part_df=tsla_df[-30:]
    fig,ax=plt.subplots()
    # 用来存放等待绘制的数据
    qutotes=[]
    # for index,(d,o,c,h,l) in enumerate(zip(tsla_part_df.index,tsla_part_df.open,tsla_part_df.close,tsla_part_df.high,tsla_part_df.low)):
    #     # 把日期数据从日期格式转换为数字格式
    #     d=mpf.date2num(d)
    #     # 存储数据
    #     val=(d,o,c,h,l)
    #     qutotes.append(val)
    #利用mpf进行绘图
    mpf.candlestick2_ochl(ax,tsla_part_df.open,tsla_part_df.close,tsla_part_df.high,tsla_part_df.low,width=0.6,colorup=__colorup__,colordown=__colordown__)
    ax.autoscale_view()
    # 使用这个方法将x轴的数据改成时间已经不行了
    # ax.xaxis_date(tsla_part_df.index.values.tolist())
    # 需要这样来进行处理
    ax.set_xticks(range(0, len(tsla_part_df['date']), 5))
    ax.set_xticklabels(tsla_part_df['date'][::5])
    plt.show()
