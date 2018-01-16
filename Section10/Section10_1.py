'''
    项目构建在虚拟的股票数据上
'''

g_with_date_week_noise = False

from abupy import ABuSymbolPd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def _gen_another_word_price(kl_another_word):
    """
    生成股票在另一个世界中的价格
    :param kl_another_word:
    :return:
    """
    for ind in np.arange(2, kl_another_word.shape[0]):
        # 前天数据
        bf_yesterday = kl_another_word.iloc[ind - 2]
        # 昨天
        yesterday = kl_another_word.iloc[ind - 1]
        # 今天
        today = kl_another_word.iloc[ind]
        # 生成今天的收盘价格
        kl_another_word.close[ind] = _gen_another_word_price_rule(
            yesterday.close, yesterday.volume,
            bf_yesterday.close, bf_yesterday.volume,
            today.volume, today.date_week)


def _gen_another_word_price_rule(yesterday_close, yesterday_volume,
                                 bf_yesterday_close,
                                 bf_yesterday_volume,
                                 today_volume, date_week):
    """
        通过前天收盘量价，昨天收盘量价，今天的量，构建另一个世界中的价格模型
    """
    # 昨天收盘价格与前天收盘价格的价格差
    price_change = yesterday_close - bf_yesterday_close
    # 昨天成交量与前天成交量的量差
    volume_change = yesterday_volume - bf_yesterday_volume

    # 如果量和价变动一致，今天价格涨，否则跌
    # 即量价齐涨－>涨, 量价齐跌－>涨，量价不一致－>跌
    sign = 1.0 if price_change * volume_change > 0 else -1.0

    # 通过date_week生成噪音，否则之后分类100%分对
    if g_with_date_week_noise:
        # 针对sign生成噪音，噪音的生效的先决条件是今天的量是这三天最大的
        gen_noise = today_volume > np.max(
            [yesterday_volume, bf_yesterday_volume])
        # 如果量是这三天最大 且是周五，下跌
        if gen_noise and date_week == 4:
            sign = -1.0
        # 如果量是这三天最大，如果是周一，上涨
        elif gen_noise and date_week == 0:
            sign = 1.0

    # 今天的涨跌幅度基础是price_change（昨天前天的价格变动）
    price_base = abs(price_change)
    # 今天的涨跌幅度变动因素：量比，
    # 今天的成交量/昨天的成交量 和 今天的成交量/前天的成交量 的均值
    price_factor = np.mean([today_volume / yesterday_volume,
                            today_volume / bf_yesterday_volume])

    if abs(price_base * price_factor) < yesterday_close * 0.10:
        # 如果 量比 * price_base 没超过10%，今天价格计算
        today_price = yesterday_close + \
                      sign * price_base * price_factor
    else:
        # 如果涨跌幅度超过10%，限制上限，下限为10%
        today_price = yesterday_close + sign * yesterday_close * 0.10
    return today_price


def change_real_to_another_word(symbol):
    """
    将原始真正的股票数据价格列只保留前两天数据，成交量，周几列完全保留
    价格列其他数据使用_gen_another_word_price变成另一个世界价格
    :param symbol:
    :return:
    """
    kl_pd = ABuSymbolPd.make_kl_df(symbol)
    if kl_pd is not None:
        # 原始股票数据也只保留价格，周几，成交量
        kl_pig_three = kl_pd.filter(['close', 'date_week', 'volume'])
        # 只保留原始头两天的交易收盘价格，其他的的都赋予nan
        kl_pig_three['close'][2:] = np.nan
        # 将其他nan价格变成猪老三世界中价格使用_gen_another_word_price
        _gen_another_word_price(kl_pig_three)
        return kl_pig_three


def another_real_show(choice_symbols, another_word_dict, real_dict):
    '''
        可视化真/假股票数据
    :return:
    '''
    import itertools
    # 4 ＊ 2
    _, axs = plt.subplots(nrows=4, ncols=2, figsize=(20, 15))
    # 将画布序列拉平
    axs_list = list(itertools.chain.from_iterable(axs))

    for symbol, ax in zip(choice_symbols, axs_list):
        # 绘制猪老三世界的股价走势
        another_word_dict[symbol].close.plot(ax=ax)
        # 同样的股票在真实世界的股价走势
        real_dict[symbol].close.plot(ax=ax)
        ax.set_title(symbol)


def gen_pig_three_feature(kl_another_word):
    '''
        生成一些会影响当天股票价格/涨跌的参数
        比如前天的价格，前天的涨跌幅等等
    :param kl_another_word:
    :return:
    '''
    # y值使用close.pct_change即涨跌幅度
    kl_another_word['regress_y'] = kl_another_word.close.pct_change()
    # 前天收盘价格
    kl_another_word['bf_yesterday_close'] = 0
    # 昨天收盘价格
    kl_another_word['yesterday_close'] = 0
    # 昨天收盘成交量
    kl_another_word['yesterday_volume'] = 0
    # 前天收盘成交量
    kl_another_word['bf_yesterday_volume'] = 0

    # 对齐特征，前天收盘价格即与今天的收盘错2个时间单位，[2:] = [:-2]
    kl_another_word['bf_yesterday_close'][2:] = \
        kl_another_word['close'][:-2]
    # 对齐特征，前天成交量
    kl_another_word['bf_yesterday_volume'][2:] = \
        kl_another_word['volume'][:-2]
    # 对齐特征，昨天收盘价与今天的收盘错1个时间单位，[1:] = [:-1]
    kl_another_word['yesterday_close'][1:] = \
        kl_another_word['close'][:-1]
    # 对齐特征，昨天成交量
    kl_another_word['yesterday_volume'][1:] = \
        kl_another_word['volume'][:-1]

    # 特征1: 价格差
    kl_another_word['feature_price_change'] = \
        kl_another_word['yesterday_close'] - \
        kl_another_word['bf_yesterday_close']

    # 特征2: 成交量差
    kl_another_word['feature_volume_Change'] = \
        kl_another_word['yesterday_volume'] - \
        kl_another_word['bf_yesterday_volume']

    # 特征3: 涨跌sign
    kl_another_word['feature_sign'] = np.sign(
        kl_another_word['feature_price_change'] * kl_another_word[
            'feature_volume_Change'])

    # 特征4: 周几
    kl_another_word['feature_date_week'] = kl_another_word[
        'date_week']

    """
        构建噪音特征, 因为猪老三也不可能全部分析正确真实的特征因素
        这里引入一些噪音特征
    """
    # 成交量乘积
    kl_another_word['feature_volume_noise'] = \
        kl_another_word['yesterday_volume'] * \
        kl_another_word['bf_yesterday_volume']

    # 价格乘积
    kl_another_word['feature_price_noise'] = \
        kl_another_word['yesterday_close'] * \
        kl_another_word['bf_yesterday_close']

    # 将数据标准化
    scaler = preprocessing.StandardScaler()
    kl_another_word['feature_price_change'] = scaler.fit_transform(
        kl_another_word['feature_price_change'].values.reshape(-1, 1))
    kl_another_word['feature_volume_Change'] = scaler.fit_transform(
        kl_another_word['feature_volume_Change'].values.reshape(-1, 1))
    kl_another_word['feature_volume_noise'] = scaler.fit_transform(
        kl_another_word['feature_volume_noise'].values.reshape(-1, 1))
    kl_another_word['feature_price_noise'] = scaler.fit_transform(
        kl_another_word['feature_price_noise'].values.reshape(-1, 1))

    # 只筛选feature_开头的特征和regress_y，抛弃前两天数据，即[2:]
    kl_pig_three_feature = kl_another_word.filter(
        regex='regress_y|feature_*')[2:]
    return kl_pig_three_feature


def gen_feature_from_symbol(symbol):
    """
    封装由一个symbol转换为特征矩阵序列函数
    :param symbol:
    :return:
    """
    # 真实世界走势数据转换到老三的世界
    kl_another_word = change_real_to_another_word(symbol)
    # 由走势转换为特征dataframe通过gen_pig_three_feature
    kl_another_word_feature_test = \
        gen_pig_three_feature(kl_another_word)

    # 构建训练的特征和label
    train_x = kl_another_word_feature_test.drop(['regress_y'], axis=1)
    train_y = pd.DataFrame(kl_another_word_feature_test['regress_y'])

    train_y.fillna(train_y.mean(), inplace=True)
    train_y_classification = train_y.copy()
    train_y_classification.loc[train_y_classification['regress_y'] > 0, 'regress_y'] = 1
    train_y_classification.loc[train_y_classification['regress_y'] <= 0, 'regress_y'] = 0

    return train_x, train_y, train_y_classification, \
           kl_another_word_feature_test


def regress_process(estimator, train_x, train_y_regress, test_x,
                    test_y_regress):
    '''
    这里是回归问题
    :param estimator:
    :param train_x:
    :param train_y_regress:
    :param test_x:
    :param test_y_regress:
    :return:
    '''
    # 用数据开始训练
    estimator.fit(train_x, train_y)
    # 用模型开始预测
    test_y_prdict_regress = estimator.predict(test_x)

    # 绘制结果
    # 绘制usFB实际股价涨跌幅度
    plt.plot(np.arange(0,len(test_y_regress)),test_y_regress.regress_y.cumsum())
    # 绘制通过模型预测的usFB股价涨跌幅度
    plt.plot(test_y_prdict_regress.cumsum())

    # 做一下交叉验证
    scores = cross_validation.cross_val_score(estimator, train_x, train_y_regress, cv=10, scoring='mean_squared_error')
    mean_sc = np.mean(np.sqrt(-scores))
    print(mean_sc)

def classification_procss(estimator,train_x,train_y_classification,test_x,test_y_classification):
    # 训练模型
    estimator.fit(train_x,train_y_classification)
    # 预测
    print(test_x.head())
    test_y_prdict_classification=estimator.predict(test_x)

    # 通过metrics.accuracy_score度量预测涨跌的准确率
    print("{} accuracy = {:.2f}".format(
        estimator.__class__.__name__,
        metrics.accuracy_score(test_y_classification,
                               test_y_prdict_classification)))

    # 针对训练集数据做交叉验证scoring='accuracy'，cv＝10
    # scores = cross_validation.cross_val_score(estimator, train_x,
    #                          train_y_classification,
    #                          cv=10,
    #                          scoring='accuracy')
    # 所有交叉验证的分数取平均值
    # mean_sc = np.mean(scores)
    # print('cross validation accuracy mean: {:.2f}'.format(mean_sc))

if __name__ == '__main__':
    '''
        上面的：
            _gen_another_word_price(kl_another_word):
            _gen_another_word_price_rule(。。。）
            change_real_to_another_word(symbol):
        都是在生成假的数据而已。
    '''
    # choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG',
                      # 'usTSLA', 'usWUBA', 'usVIPS']
    choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU','usGOOG']
    another_word_dict = {}
    real_dict = {}
    for symbol in choice_symbols:
        # 猪老三世界的股票走势字典
        another_word_dict[symbol] = change_real_to_another_word(symbol)
        # 真实世界的股票走势字典，这里不考虑运行效率问题
        real_dict[symbol] = ABuSymbolPd.make_kl_df(symbol)
    # 表10-1所示,每支股票大概500个数据
    print(another_word_dict['usNOAH'].head())
    # 创建特征的时候，可以用shift，而不用上面函数里那么[1:]=[:-1]的写法
    # print(another_word_dict['usNOAH'].volume.shift(1).head())

    # 通过原始数据的组合得到新的特征
    pig_three_feature = None
    for symbol in choice_symbols:
        kl_another_word = another_word_dict[symbol]
        kl_feature = gen_pig_three_feature(kl_another_word)
        pig_three_feature = kl_another_word if pig_three_feature is None else pig_three_feature.append(kl_feature)
    print(pig_three_feature.shape)

    features_pig_three = pig_three_feature.fillna(0)
    features_pig_three = features_pig_three.filter(
        ['regress_y', 'feature_price_change', 'feature_volume_Change', 'feature_sign', 'feature_date_week',
         'feature_volume_noise', 'feature_price_noise'])

    # 构建训练的特征和label
    train_x = features_pig_three.drop(['regress_y'], axis=1)
    train_y = pd.DataFrame(pig_three_feature['regress_y'])
    train_y.fillna(train_y.mean(), inplace=True)
    train_y_classification = train_y.copy()
    train_y_classification.loc[train_y_classification['regress_y'] > 0, 'regress_y'] = 1
    train_y_classification.loc[train_y_classification['regress_y'] <= 0, 'regress_y'] = 0

    print(train_y.head())
    print(train_x.head())
    print(train_y_classification.head())

    # 拿到测试数据
    test_x, test_y_regress, test_y_classification, \
    kl_another_word_feature_test = gen_feature_from_symbol('usFB')

    print(train_x.shape)
    print(test_x.shape)

    # 调用，开始做回归
    # estimator = LinearRegression()
    # estimator=AdaBoostRegressor(n_estimators=100)
    # estimator=RandomForestRegressor(n_estimators=100)
    # regress_process(estimator, train_x, train_y, test_x, test_y_regress)
    # plt.show()

    # 调用，开始做分类
    # estimator=LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
    estimator=SVC(kernel='rbf')
    classification_procss(estimator,train_x.head(200),train_y_classification.head(200),test_x,test_y_classification)


