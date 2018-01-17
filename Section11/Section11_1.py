from abupy import AbuFactorAtrNStop, AbuFactorPreAtrNStop, AbuFactorCloseAtrNStop, AbuFactorBuyBreak
from abupy import abu, EMarketTargetType, AbuMetricsBase, ABuMarketDrawing, ABuProgress, ABuSymbolPd
from abupy import EMarketTargetType, EDataCacheType, EMarketSourceType, EMarketDataFetchMode, EStoreAbu, AbuUmpMainMul
from abupy import AbuUmpMainDeg, AbuUmpMainJump, AbuUmpMainPrice, AbuUmpMainWave, feature, AbuFeatureDegExtend
from abupy import AbuUmpEdgeDeg, AbuUmpEdgePrice, AbuUmpEdgeWave, AbuUmpEdgeFull, AbuUmpEdgeMul, AbuUmpEegeDegExtend
from abupy import AbuUmpMainDegExtend, ump, Parallel, delayed, AbuMulPidProgress, AbuProgress

if __name__ == '__main__':
    abu_result_tuple_train = abu.load_abu_result_tuple(n_folds=5, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                                       custom_name='train_cn')
    abu_result_tuple_test = abu.load_abu_result_tuple(n_folds=5, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                                      custom_name='test_cn')
    ABuProgress.clear_output()
    print('训练集结果：')
    metrics_train = AbuMetricsBase.show_general(*abu_result_tuple_train, only_show_returns=True)
    print('测试集结果：')
    metrics_test = AbuMetricsBase.show_general(*abu_result_tuple_test, only_show_returns=True)